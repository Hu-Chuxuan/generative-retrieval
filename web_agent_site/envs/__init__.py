from gym.envs.registration import register

try:
    from web_agent_site.envs.web_agent_site_env import WebAgentSiteEnv
    register(
        id='WebAgentSiteEnv-v0',
        entry_point='web_agent_site.envs:WebAgentSiteEnv',
    )
except ImportError:
    pass

from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

register(
  id='WebAgentTextEnv-v0',
  entry_point='web_agent_site.envs:WebAgentTextEnv',
)