"""Git History:第二个 ContextProvider(review region 的真 body;Phase 6)。

git log 子进程召回 + 关键词搜索。provider 把 GitEvent 包成 ContextBlock(framing=data),
经 ProviderRegistry 注册为 ``"git"``(server bootstrap);skill ``git-recall`` 的 ref 指向它。

import 无 git 副作用(``GitStore.list_commits`` 在 retrieve 时才跑子进程)。
"""

from . import store
from .base import GitEvent
from .provider import GitProvider

__all__ = ["GitEvent", "GitProvider", "store"]
