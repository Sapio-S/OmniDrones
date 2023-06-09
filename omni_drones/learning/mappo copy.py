import time
from typing import Any, Dict, List, Optional, Tuple, Union

import functorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchopt
from tensordict import TensorDict
from tensordict.nn import make_functional, TensorDictModule
from torch.optim import lr_scheduler
from torchrl.data import CompositeSpec

from omni_drones.envs.isaac_env import AgentSpec

from .utils import valuenorm
from .utils.clip_grad import clip_grad_norm_
from .utils.gae import compute_gae

LR_SCHEDULER = lr_scheduler._LRScheduler


class MAPPOPolicy(nn.Module):
    def __init__(
        self, cfg, agent_spec: AgentSpec, act_name: str = None, device="cuda"
    ) -> None:
        super().__init__()

        self.cfg = cfg
        self.agent_spec = agent_spec
        self.device = device

        self.clip_param = cfg.clip_param
        self.ppo_epoch = int(cfg.ppo_epochs)
        self.num_minibatches = int(cfg.num_minibatches)
        self.normalize_advantages = cfg.normalize_advantages

        self.entropy_coef = cfg.entropy_coef
        self.gae_gamma = cfg.gamma
        self.gae_lambda = cfg.gae_lambda

        self.act_dim = agent_spec.action_spec.shape.numel()

        if cfg.reward_weights is not None:
            self.reward_weights = torch.as_tensor(cfg.reward_weights, device=device)
        else:
            self.reward_weights = torch.ones(
                self.agent_spec.reward_spec.shape, device=device
            )

        self.obs_name = f"{self.agent_spec.name}.obs"
        self.act_name = act_name
        self.state_name = f"{self.agent_spec.name}.state"
        self.reward_name = f"{self.agent_spec.name}.reward"

        self.make_actor()
        self.make_critic()

        self.train_in_keys = list(set(
            self.actor_in_keys
            + self.actor_out_keys
            + self.critic_in_keys
            + self.critic_out_keys
            + [
                "next",
                self.act_logps_name,
                ("reward", self.reward_name),
                "state_value",
            ]
        ))

        self.in_keys = [self.obs_name, self.state_name]
        self.n_updates = 0

    @property
    def act_logps_name(self):
        return f"{self.agent_spec.name}.action_logp"

    def make_actor(self):
        cfg = self.cfg.actor

        self.actor_in_keys = [self.obs_name, self.act_name]
        self.actor_out_keys = [
            self.act_name,
            self.act_logps_name,
            f"{self.agent_spec.name}.action_entropy",
        ]
        self.actor_opt = torchopt.adam(lr=cfg.lr)

        def actor_loss(params, actor_input, advantages, log_probs_old):
            actor_output = fmodel(params, buffers, actor_input)
            log_probs = actor_output[self.act_logps_name]
            dist_entropy = actor_output[f"{self.agent_spec.name}.action_entropy"]
            assert advantages.shape == log_probs.shape == dist_entropy.shape

            ratio = torch.exp(log_probs - log_probs_old)
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                * advantages
            )

            policy_loss = -torch.min(surr1, surr2) * self.act_dim
            policy_loss = torch.mean(policy_loss - self.entropy_coef * dist_entropy)

            log_ratio = log_probs - log_probs_old
            approx_kl = torch.mean(torch.exp(log_ratio) - 1 - log_ratio)
            clip_frac = torch.mean((torch.abs(ratio - 1) > self.clip_param).float())

            return policy_loss, (dist_entropy, approx_kl, clip_frac)

        if self.cfg.share_actor:
            actor = make_ppo_actor(
                cfg, self.agent_spec.observation_spec, self.agent_spec.action_spec
            )
            if actor.rnn:
                self.actor_in_keys.extend(
                    [f"{self.agent_spec.name}.rnn_state", "is_init"]
                )
                self.actor_out_keys.append(f"{self.agent_spec.name}.rnn_state")
                self.minibatch_seq_len = self.cfg.actor.rnn.train_seq_len
                assert self.minibatch_seq_len <= self.cfg.train_every

            self.actor = TensorDictModule(
                actor,
                in_keys=self.actor_in_keys,
                out_keys=self.actor_out_keys,
            ).to(self.device)
            self.actor_func_module = (
                fmodel,
                params,
                buffers,
            ) = functorch.make_functional_with_buffers(self.actor)
            self.actor_func = functorch.vmap(
                fmodel, in_dims=(None, None, 1), out_dims=1, randomness="different"
            )
            self.actor_opt_state = self.actor_opt.init(params)
            self.actor_loss = functorch.vmap(actor_loss, in_dims=(None, 1, 1, 1))
        else:
            self.actor = nn.ModuleList(
                [
                    TensorDictModule(
                        make_ppo_actor(
                            cfg,
                            self.agent_spec.observation_spec,
                            self.agent_spec.action_spec,
                        ),
                        in_keys=self.actor_in_keys,
                        out_keys=self.actor_out_keys,
                    )
                    for _ in range(self.agent_spec.n)
                ]
            ).to(self.device)
            self.actor_func_module = (
                fmodel,
                params,
                buffers,
            ) = functorch.combine_state_for_ensemble(self.actor)
            self.actor_func = functorch.vmap(
                fmodel, in_dims=(0, 0, 1), out_dims=1, randomness="different"
            )
            self.actor_opt_state = functorch.vmap(self.actor_opt.init)(params)
            for param in params:
                param.requires_grad_(True)
            self.actor_loss = functorch.vmap(actor_loss, in_dims=(0, 1, 1, 1))

    def make_critic(self):
        cfg = self.cfg.critic

        if cfg.use_huber_loss:
            self.critic_loss_fn = nn.HuberLoss(reduction="none", delta=cfg.huber_delta)
        else:
            self.critic_loss_fn = nn.MSELoss(reduction="none")

        if self.cfg.critic_input == "state":
            if self.agent_spec.state_spec is None:
                raise ValueError
            self.critic_in_keys = [f"{self.agent_spec.name}.state"]
            self.critic_out_keys = ["state_value"]

            self.critic = TensorDictModule(
                CentralizedCritic(
                    cfg,
                    entity_ids=torch.arange(self.agent_spec.n, device=self.device),
                    state_spec=self.agent_spec.state_spec,
                    reward_spec=self.agent_spec.reward_spec,
                ),
                in_keys=self.critic_in_keys,
                out_keys=self.critic_out_keys,
            ).to(self.device)
            self.value_func = self.critic

        elif self.cfg.critic_input == "obs":
            self.critic_in_keys = [f"{self.agent_spec.name}.obs"]
            self.critic_out_keys = ["state_value"]

            self.critic = TensorDictModule(
                SharedCritic(
                    cfg, self.agent_spec.observation_spec, self.agent_spec.reward_spec
                ),
                in_keys=self.critic_in_keys,
                out_keys=self.critic_out_keys,
            ).to(self.device)
            self.value_func = functorch.vmap(self.critic, in_dims=1, out_dims=1)
        else:
            raise ValueError(self.cfg.critic_input)

        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scheduler = cfg.lr_scheduler
        if scheduler is not None:
            scheduler = eval(scheduler)
            self.critic_opt_scheduler: LR_SCHEDULER = scheduler(
                self.critic_opt, **cfg.lr_scheduler_kwargs
            )

        if hasattr(cfg, "value_norm") and cfg.value_norm is not None:
            # The original MAPPO implementation uses ValueNorm1 with a very large beta,
            # and normalizes advantages at batch level.
            # Tianshou (https://github.com/thu-ml/tianshou) uses ValueNorm2 with subtract_mean=False,
            # and normalizes advantages at mini-batch level.
            # Empirically the performance is similar on most of the tasks.
            cls = getattr(valuenorm, cfg.value_norm["class"])
            self.value_normalizer: valuenorm.Normalizer = cls(
                input_shape=self.agent_spec.reward_spec.shape,
                **cfg.value_norm["kwargs"],
            ).to(self.device)

    def policy_op(self, tensordict: TensorDict, training: bool):
        fmodel, params, buffers = self.actor_func_module
        actor_input = tensordict.select(*self.actor_in_keys, strict=False)
        actor_input.batch_size = [*actor_input.batch_size, self.agent_spec.n]
        tensordict = self.actor_func(
            params, buffers, actor_input, deterministic=(not training)
        )
        return tensordict

    def value_op(self, tensordict: TensorDict) -> TensorDict:
        critic_input = tensordict.select(*self.critic_in_keys)
        critic_input.batch_size = [*critic_input.batch_size, self.agent_spec.n]
        tensordict = self.value_func(critic_input)
        return tensordict

    def __call__(self, tensordict: TensorDict):
        input_td = tensordict.select(*self.in_keys, strict=False)
        tensordict.update(self.policy_op(input_td, True))
        tensordict.update(self.value_op(input_td))
        return tensordict

    def update_actor(self, batch: TensorDict) -> Dict[str, Any]:
        fmodel, params, buffers = self.actor_func_module
        advantages = batch["advantages"]
        actor_input = batch.select(*self.actor_in_keys)
        actor_input.batch_size = [*actor_input.batch_size, self.agent_spec.n]
        policy_loss, (dist_entropy, approx_kl, clip_frac) = self.actor_loss(
            params, actor_input, advantages, batch[self.act_logps_name]
        )
        if self.cfg.share_actor:
            grads = torch.autograd.grad(policy_loss.mean(), params)
            grad_norm = clip_grad_norm_(grads, max_norm=self.cfg.max_grad_norm)
            updates, self.actor_opt_state = self.actor_opt.update(
                grads, self.actor_opt_state
            )
        else:
            grads = torch.autograd.grad(policy_loss.sum(), params)
            grad_norm = functorch.vmap(clip_grad_norm_)(
                grads, max_norm=self.cfg.max_grad_norm
            )
            updates, self.actor_opt_state = functorch.vmap(self.actor_opt.update)(
                grads, self.actor_opt_state
            )
        torchopt.apply_updates(params, updates, inplace=True)

        return {
            "policy_loss": policy_loss.mean(),
            "actor_grad_norm": grad_norm.mean(),
            "dist_entropy": dist_entropy.mean(),
            "clip_fraction": clip_frac.mean(),
            "approx_kl": approx_kl.mean(),
        }

    def update_critic(self, batch: TensorDict) -> Dict[str, Any]:
        critic_input = batch.select(*self.critic_in_keys)
        values = self.value_op(critic_input)["state_value"]
        b_values = batch["state_value"]
        b_returns = batch["returns"]
        assert values.shape == b_values.shape == b_returns.shape
        value_pred_clipped = b_values + (values - b_values).clamp(
            -self.clip_param, self.clip_param
        )

        value_loss_clipped = self.critic_loss_fn(b_returns, value_pred_clipped)
        value_loss_original = self.critic_loss_fn(b_returns, values)

        value_loss = torch.max(value_loss_original, value_loss_clipped)

        value_loss.sum(-1).mean().backward()  # do not multiply weights here
        grad_norm = nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.cfg.max_grad_norm
        )
        self.critic_opt.step()
        self.critic_opt.zero_grad(set_to_none=True)
        return {
            "value_loss": value_loss.mean(),
            "critic_grad_norm": grad_norm,
        }

    def _get_dones(self, tensordict: TensorDict):
        env_done = tensordict[("next", "done")].unsqueeze(-1)
        agent_done = tensordict.get(
            ("next", f"{self.agent_spec.name}.done"),
            env_done.expand(*env_done.shape[:-2], self.agent_spec.n, 1),
        )
        done = agent_done | env_done
        return done

    def train_op(self, tensordict: TensorDict):
        tensordict = tensordict.select(*self.train_in_keys, strict=False)
        next_tensordict = tensordict["next"][:, -1]
        with torch.no_grad():
            value_output = self.value_op(next_tensordict)

        rewards = tensordict.get(("next", "reward", f"{self.agent_spec.name}.reward"))

        values = tensordict["state_value"]
        next_value = value_output["state_value"].squeeze(0)

        if hasattr(self, "value_normalizer"):
            values = self.value_normalizer.denormalize(values)
            next_value = self.value_normalizer.denormalize(next_value)

        dones = self._get_dones(tensordict)

        tensordict["advantages"], tensordict["returns"] = compute_gae(
            rewards,
            dones,
            values,
            next_value,
            gamma=self.gae_gamma,
            lmbda=self.gae_lambda,
        )

        advantages_mean = tensordict["advantages"].mean()
        advantages_std = tensordict["advantages"].std()
        if self.normalize_advantages:
            tensordict["advantages"] = (tensordict["advantages"] - advantages_mean) / (
                advantages_std + 1e-8
            )

        if hasattr(self, "value_normalizer"):
            self.value_normalizer.update(tensordict["returns"])
            tensordict["returns"] = self.value_normalizer.normalize(
                tensordict["returns"]
            )

        train_info = []
        for ppo_epoch in range(self.ppo_epoch):
            dataset = make_dataset_naive(tensordict, self.cfg.num_minibatches, self.minibatch_seq_len if hasattr(self, "minibatch_seq_len") else 1)
            for minibatch in dataset:
                train_info.append(
                    TensorDict(
                        {
                            **self.update_actor(minibatch),
                            **self.update_critic(minibatch),
                        },
                        batch_size=[],
                    )
                )

        train_info: TensorDict = torch.stack(train_info)
        train_info = train_info.apply(lambda x: x.mean(0), batch_size=[])
        train_info["advantages_mean"] = advantages_mean
        train_info["advantages_std"] = advantages_std
        train_info["action_norm"] = (
            tensordict[self.act_name].float().norm(dim=-1).mean()
        )
        if hasattr(self, "value_normalizer"):
            train_info["value_running_mean"] = self.value_normalizer.running_mean.mean()
        self.n_updates += 1
        return {f"{self.agent_spec.name}/{k}": v for k, v in train_info.items()}


def make_dataset_naive(
    tensordict: TensorDict, num_minibatches: int = 4, seq_len: int = 1
):
    if seq_len > 1:
        N, T = tensordict.shape
        T = (T // seq_len) * seq_len
        tensordict = tensordict[:, :T].reshape(-1, seq_len)
        perm = torch.randperm(
            (tensordict.shape[0] // num_minibatches) * num_minibatches,
            device=tensordict.device,
        ).reshape(num_minibatches, -1)
        for indices in perm:
            yield tensordict[indices]
    else:
        tensordict = tensordict.reshape(-1)
        perm = torch.randperm(
            (tensordict.shape[0] // num_minibatches) * num_minibatches,
            device=tensordict.device,
        ).reshape(num_minibatches, -1)
        for indices in perm:
            yield tensordict[indices]


from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    MultiDiscreteTensorSpec,
    TensorSpec,
    UnboundedContinuousTensorSpec as UnboundedTensorSpec,
)

from .utils.distributions import (
    DiagGaussian,
    IndependentBetaModule,
    IndependentNormalModule,
    MultiCategoricalModule,
)
from .utils.network import (
    ENCODERS_MAP,
    MLP,
    SplitEmbedding,
)

from .modules.rnn import GRU


def make_ppo_actor(cfg, observation_spec: TensorSpec, action_spec: TensorSpec):
    if isinstance(observation_spec, (BoundedTensorSpec, UnboundedTensorSpec)):
        if not len(observation_spec.shape) == 1:
            raise ValueError
        input_dim = observation_spec.shape[0]
        base = nn.Sequential(
            nn.LayerNorm(input_dim),
            MLP([input_dim] + cfg.hidden_units),
        )
        base.output_shape = torch.Size((cfg.hidden_units[-1],))
    elif isinstance(observation_spec, CompositeSpec):
        encoder_cls = ENCODERS_MAP[cfg.attn_encoder]
        base = encoder_cls(observation_spec)
    else:
        raise NotImplementedError(observation_spec)

    if isinstance(action_spec, MultiDiscreteTensorSpec):
        act_dist = MultiCategoricalModule(action_spec.nvec)
    elif isinstance(action_spec, (UnboundedTensorSpec, BoundedTensorSpec)):
        action_dim = action_spec.shape.numel()
        act_dist = DiagGaussian(base.output_shape.numel(), action_dim, True, 0.01)
        # act_dist = IndependentNormalModule(inputs_dim, action_dim, True)
        # act_dist = IndependentBetaModule(inputs_dim, action_dim)
    else:
        raise NotImplementedError(action_spec)

    if cfg.get("rnn", None):
        rnn_cls = {"gru": GRU}[cfg.rnn.cls.lower()]
        rnn = rnn_cls(input_size=base.output_shape.numel(), **cfg.rnn.kwargs)
    else:
        rnn = None

    return Actor(base, act_dist, rnn)


class Actor(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        act_dist: nn.Module,
        rnn: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.act_dist = act_dist
        self.rnn = rnn

    def forward(
        self,
        obs: Union[torch.Tensor, TensorDict],
        action: torch.Tensor = None,
        is_init=None,
        rnn_state=None,
        deterministic=False,
    ):
        actor_features = self.encoder(obs)
        if self.rnn is not None:
            if actor_features.dim() == 2:  # single-step inference
                actor_features, rnn_state = self.rnn(
                    actor_features.unsqueeze(0), rnn_state, is_init
                )
                actor_features, rnn_state = actor_features.squeeze(
                    0
                ), rnn_state.squeeze(0)
            else:  # multi-step re-rollout during training
                actor_features, rnn_state = self.rnn(
                    actor_features,
                    rnn_state[0] if rnn_state is not None else None,
                    is_init,
                )
        else:
            rnn_state = None
        action_dist = self.act_dist(actor_features)

        if action is None:
            action = action_dist.mode if deterministic else action_dist.sample()
            action_log_probs = action_dist.log_prob(action).unsqueeze(-1)
            return action, action_log_probs
        else:
            action_log_probs = action_dist.log_prob(action).unsqueeze(-1)
            dist_entropy = action_dist.entropy().unsqueeze(-1)
            return action, action_log_probs, dist_entropy

    def evaluate(
        self,
        obs: Union[torch.Tensor, TensorDict],
        action: torch.Tensor = None,
        done=None,
        rnn_state=None,
        deterministic=False,
    ):
        ...


class SharedCritic(nn.Module):
    def __init__(
        self,
        args,
        observation_spec: TensorSpec,
        reward_spec: TensorSpec,
    ):
        super().__init__()
        if isinstance(observation_spec, (BoundedTensorSpec, UnboundedTensorSpec)):
            input_dim = observation_spec.shape.numel()
            self.base = nn.Sequential(
                nn.LayerNorm(input_dim), MLP([input_dim] + args.hidden_units)
            )
            self.base.output_shape = torch.Size((args.hidden_units[-1],))
        elif isinstance(observation_spec, CompositeSpec):
            encoder_cls = ENCODERS_MAP[args.attn_encoder]
            self.base = encoder_cls(observation_spec)
        else:
            raise TypeError(observation_spec)

        self.output_shape = reward_spec.shape
        self.v_out = nn.Linear(
            self.base.output_shape.numel(), self.output_shape.numel()
        )
        nn.init.orthogonal_(self.v_out.weight, args.gain)

    def forward(self, obs: torch.Tensor):
        critic_features = self.base(obs)
        values = self.v_out(critic_features)
        if len(self.output_shape) > 1:
            values = values.unflatten(-1, self.output_shape)
        return values


INDEX_TYPE = Union[int, slice, torch.LongTensor, List[int]]


class CentralizedCritic(nn.Module):
    """Critic for centralized training.

    Args:
        entity_ids: indices of the entities that are considered as agents.

    """

    def __init__(
        self,
        cfg,
        entity_ids: INDEX_TYPE,
        state_spec: CompositeSpec,
        reward_spec: TensorSpec,
        embed_dim=128,
        nhead=1,
        num_layers=1,
    ):
        super().__init__()
        self.entity_ids = entity_ids
        self.embed = SplitEmbedding(state_spec, embed_dim=embed_dim)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=embed_dim,
                dropout=0.0,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.output_shape = reward_spec.shape
        self.v_out = nn.Linear(embed_dim, self.output_shape.shape.numel())

    def forward(self, x: torch.Tensor):
        x = self.embed(x)
        x = self.encoder(x)
        values = self.v_out(x[..., self.entity_ids, :]).unflatten(-1, self.output_shape)
        return values
