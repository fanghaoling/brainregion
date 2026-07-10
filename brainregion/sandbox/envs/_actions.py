"""Phase 4.8 集中动作模型 —— 消 4 份 ``_ACTION_DELTA`` mirror(gridworld / memory_region /
topo_region / env_eval 各一份)。

GPT#4 升级 dict→``ActionModel`` 类(mode/vocab/delta/heading_after/is_turn),为 hex/continuous/
diagonal 留扩展点(不预做子类,YAGNI)。**纯算术常量,不依赖 env 类** → import 不违 D.2「自给」
(D.2 禁读 env._agent/render,非共享动作算术)。review opus-8:所有 heading 变换**唯一**走
``ActionModel.heading_after``(防左右旋约定分叉重演 mirror)。

两模式:
- ``abs``:世界绝对方向(up/down/left/right → 固定 dx,dy;无 heading 概念)。GridWorld 默认(零回归)。
- ``ego``:自我中心(forward 沿 heading 走;turn_left/right 改 heading)。GridWorld(ego_actions=True)。

判模式用 ``env.ego_actions`` flag(GPT#1),**不用** ``hasattr(env,"_heading")`` —— heading+abs 可能是
合法组合(有 heading 但 abs 动作),不能把「有 heading」等价「ego」。
"""
from __future__ import annotations

# 世界绝对方向 → (dx, dy)。x=列(右+),y=行(下+),原点左上 (0,0)。镜像原 4 份 _ACTION_DELTA。
ABS_DELTA: dict[str, tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}

# 4 朝向 → forward 的 (dx, dy)(沿此走)。
HEADING_DELTA: dict[str, tuple[int, int]] = {
    "N": (0, -1),
    "E": (1, 0),
    "S": (0, 1),
    "W": (-1, 0),
}

# 左转 / 右转(90°;封闭映射 → 输出恒 in {N,E,S,W},review opus-4 heading 合法性)。
TURN_LEFT: dict[str, str] = {"N": "W", "W": "S", "S": "E", "E": "N"}
TURN_RIGHT: dict[str, str] = {"N": "E", "E": "S", "S": "W", "W": "N"}

HEADING_ZH: dict[str, str] = {"N": "北", "E": "东", "S": "南", "W": "西"}

# abs 方向词 → heading(ego recipe 转换用;topo state 把 abs frontier/backtrack 转相对 heading)。
ABS_DIR_HEADING: dict[str, str] = {"up": "N", "down": "S", "left": "W", "right": "E"}

INITIAL_HEADING = "E"  # 固定初始朝向(避 maze 布局成混杂因子;所有 maze 一致)

ABS_VOCAB: tuple[str, ...] = tuple(ABS_DELTA.keys())           # ("up","down","left","right")
EGO_VOCAB: tuple[str, ...] = ("forward", "turn_left", "turn_right")  # turn_180 defer(review Plan agent)


class ActionModel:
    """动作解析模型(纯算术;abs / ego 两单例)。

    - ``delta(action, heading)``:forward/abs-move → ``(dx,dy)``;turn → ``None``(不位移)。
    - ``heading_after(action, heading)``:turn → 新 heading;forward/abs → 原 heading(不变)。
    - ``is_turn(action)``:ego turn 类(``True``);abs 恒 ``False``(abs 无纯转向)。
    - ``is_move(action)``:产生位移的动作(forward / abs 任意);turn/非法 → ``False``。

    heading 合法性(review opus-4):非法 heading(∉ {N,E,S,W})→ ``heading_after``/``delta`` 抛
    ``ValueError``(防 desync 脏 heading 以 KeyError 崩栈,可诊断降级)。
    """

    def __init__(self, mode: str) -> None:
        if mode not in ("abs", "ego"):
            raise ValueError(f"ActionModel mode 须 abs|ego, got {mode!r}")
        self.mode = mode
        self.vocab = EGO_VOCAB if mode == "ego" else ABS_VOCAB

    def delta(self, action: str, heading: str) -> tuple[int, int] | None:
        """action → ``(dx,dy)``;turn / 非法 → ``None``。heading 仅 ego forward 用(abs 忽略)。"""
        if self.mode == "abs":
            return ABS_DELTA.get(action)
        h = self._norm_heading(heading)
        if action == "forward":
            return HEADING_DELTA[h]
        return None  # turn_left / turn_right / 非法 → None(不位移)

    def heading_after(self, action: str, heading: str) -> str:
        """action 后的新 heading。ego turn → 旋转;forward/abs/非法 → 原 heading(不变)。"""
        if self.mode == "abs":
            return heading  # abs 不改 heading(概念不用)
        h = self._norm_heading(heading)
        if action == "turn_left":
            return TURN_LEFT[h]
        if action == "turn_right":
            return TURN_RIGHT[h]
        return h  # forward / 非法 → 不变

    def is_turn(self, action: str) -> bool:
        """ego turn 类;abs 恒 False。"""
        if self.mode == "abs":
            return False
        return action in ("turn_left", "turn_right")

    def is_move(self, action: str) -> bool:
        """产生位移的动作(forward / abs 任意合法);turn / 非法 → False。"""
        if self.mode == "abs":
            return action in ABS_DELTA
        return action == "forward"

    @staticmethod
    def _norm_heading(heading: str) -> str:
        if heading not in HEADING_DELTA:
            raise ValueError(f"非法 heading {heading!r}, 须 N|E|S|W")
        return heading


def relative_direction(abs_dir: str, heading: str) -> str:
    """abs 方向词(up/down/left/right)→ 相对 ``heading`` 的方位(forward/left/right/back)。

    ego 模式 topo 用:把 abs frontier/backtrack 转成主脑可执行的相对方位(面朝 heading 时,该 abs 方向在
    前/左/右/后)。验证:heading=E 时 left=N(左侧)、right=S、forward=E、back=W。纯算术,用 TURN_LEFT/RIGHT。
    """
    target = ABS_DIR_HEADING.get(abs_dir)
    if target is None:
        return abs_dir  # 非 abs 词(已是相对?)原样返
    h = ActionModel._norm_heading(heading)
    if target == h:
        return "forward"
    if target == TURN_LEFT[h]:
        return "left"
    if target == TURN_RIGHT[h]:
        return "right"
    return "back"


ABS = ActionModel("abs")
EGO = ActionModel("ego")


__all__ = [
    "ActionModel", "ABS", "EGO",
    "ABS_DELTA", "HEADING_DELTA", "TURN_LEFT", "TURN_RIGHT", "HEADING_ZH", "ABS_DIR_HEADING",
    "INITIAL_HEADING", "ABS_VOCAB", "EGO_VOCAB", "relative_direction",
]
