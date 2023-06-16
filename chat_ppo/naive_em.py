import torch
from .utils import compute_reward, normalize

from .base_em import Experience, ExperienceMaker


class NaiveExperienceMaker(ExperienceMaker):
    """
    Naive experience maker.
    """

    @torch.no_grad()
    def make_experience(self, input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor, **generate_kwargs) -> Experience:
        self.actor.eval()
        # self.critic.eval()
        self.initial_model.eval()
        # self.reward_model.eval()

        # sequences, attention_mask, action_mask = self.actor.generate(input_ids,
        #                                                              return_action_mask=True,
        #                                                              **generate_kwargs)
        # num_actions = action_mask.size(1)
        outputs = self.actor(input_ids)
                            # return_action_mask=True,
                            # **generate_kwargs) # sequences, attention_mask, action_mask
        # num_actions = action_mask.size(1)

        action_log_probs = self.actor(input_ids) # sequences, num_actions, attention_mask
        base_action_log_probs = self.initial_model(input_ids)
        value = self.critic(outputs['logits'], labels, attention_mask)
        r = self.reward_model(outputs['logits'], labels, attention_mask)

        reward = compute_reward(r, self.kl_coef, action_log_probs, base_action_log_probs, action_mask=action_mask)

        advantage = reward - value
        # TODO(ver217): maybe normalize adv
        if advantage.ndim == 1:
            advantage = advantage.unsqueeze(-1)

        return Experience(sequences, action_log_probs, value, reward, advantage, attention_mask, action_mask)
