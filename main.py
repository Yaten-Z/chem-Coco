import pygame
import random
import sys
import os
import cv2
import mediapipe as mp
import threading
import time
from collections import deque
import logging
import webbrowser
import re
from openai import OpenAI
from dotenv import load_dotenv
from urllib.parse import urlparse

load_dotenv()

# 配置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("kimiai")

KIMI_API_KEY = os.getenv("KIMI_API_KEY")
KIMI_BASE_URL="https://api.moonshot.cn/v1"
KIMI_MODEL = "kimi-k2-turbo-preview" # 这是一个模型名称，可以保留在代码中，或者也放到 .env 中

pygame.init()
WIDTH, HEIGHT = 1400, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chemistry Learner")

# Color
WHITE = (255, 255, 255)
BLACK = (30, 30, 30)
PRIMARY_BLUE = (30, 144, 255)
ACCENT_ORANGE = (255, 140, 0)
SUCCESS_GREEN = (46, 204, 113)
ERROR_RED = (231, 76, 60)
HOVER_YELLOW = (255, 230, 109)
BACKGROUND_LIGHT = (240, 248, 255)
BACKGROUND_DARK = (210, 220, 230)

CURSOR_COLOR = ERROR_RED
CURSOR_RADIUS = 12

# 化学物质列表
ALLOWED_SUBSTANCES_LIST = (
    "H₂, O₂, N₂, Cl₂, C, S, P, Fe, Cu, Zn, Al, Mg, Ag, Au, Hg, "
    "H₂O, CO, CO₂, CaO, Fe₂O₃, CuO, MgO, Al₂O₃, MnO₂, SO₂, SO₃, "
    "HCl, H₂SO₄, HNO₃, H₂CO₃, H₃PO₄, CH₃COOH, "
    "NaOH, Ca(OH)₂, KOH, Ba(OH)₂, Cu(OH)₂, Fe(OH)₃, Al(OH)₃, NH₃·H₂O, "
    "NaCl, CaCl₂, BaCl₂, FeCl₃, CuCl₂, AgCl, NH₄Cl, "
    "Na₂SO₄, CuSO₄·5H₂O, BaSO₄, CaSO₄·2H₂O, FeSO₄, ZnSO₄, "
    "Na₂CO₃, NaHCO₃, CaCO₃, BaCO₃, K₂CO₃, "
    "AgNO₃, KNO₃, NaNO₃, Cu(NO₃)₂, Ba(NO₃)₂, "
    "Na₃PO₄, Ca₃(PO₄)₂, NH₄H₂PO₄, "
    "FeS, CuS, ZnS, "
    "KMnO₄, K₂MnO₄, KClO₃, NaClO, "
    "H₂O₂, CH₄, C₂H₅OH, C₆H₁₂O₆, C₁₂H₂₂O₁₁, (C₆H₁₀O₅)ₙ, 蛋白质, 油脂, 石蜡, "
    "KAl(SO₄)₂·12H₂O, SiO₂, NH₃"
)
def get_font(size):
    # 1. 优先尝试本地字体文件
    font_files = [
        #"fonts/Noto Sans CJK Regular.otf",
        "fonts/Heiti TC.ttf",  # 常用黑体
        "fonts/msyh.ttc",  # 微软雅黑
        "fonts/simhei.ttf"
    ]

    for font_path in font_files:
        font_path = os.path.expanduser(font_path)
        if os.path.exists(font_path):
            try:
                font = pygame.font.Font(font_path, size)
                # 检查是否能渲染中文，以确认字体有效
                if font.render("测", True, BLACK).get_width() > 0:
                    logging.debug(f"成功加载本地中文字体: {font_path}")
                    return font
            except Exception as e:
                logging.warning(f"加载本地字体失败: {font_path}, {e}")
                continue

    # 2. 尝试系统字体名称 (思源黑体、Noto 优先级最高)
    system_font_names = [
        "Source Han Sans CN",
        "Source Han Sans SC",
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "PingFang SC",
        "SimHei",
        "SimSun",
        "Arial Unicode MS"
    ]

    for name in system_font_names:
        try:
            font = pygame.font.SysFont(name, size)
            if font.render("测", True, BLACK).get_width() > 0:
                logging.debug(f"已加载系统字体: {name}")
                return font
        except:
            continue

    # 没有字体适配：
    logging.warning("未找到支持中文的字体，使用默认Unicode字体")
    return pygame.font.SysFont(pygame.font.get_default_font(), size)


# 全局字体定义
font_small = get_font(24)
font_medium = get_font(32)
font_large = get_font(48)
font_tiny = get_font(18)  # 用于下标


def render_chemical_formula(surface, formula_text, x, y, main_font, sub_font, color):
    """
    渲染化学式，使用 get_ascent() 针对 CJK 字体进行精确下标定位。
    """
    current_x = x

    # 核心修复: 使用 get_ascent() (字符基线上方的高度) 计算偏移，忽略不稳定的行高。
    main_ascent = main_font.get_ascent()

    # 下标偏移量：设置为基线高度的 55% 左右，使得下标看起来“挂”在主字母右下角。
    subscript_offset_y = int(main_ascent * 0.55)

    # 状态标记
    is_prev_digit = False
    is_prev_subscript = False

    for i, char in enumerate(formula_text):
        use_sub_font = False

        if char.isdigit():
            # 智能判断下标逻辑
            if i > 0:
                prev_char = formula_text[i - 1]
                # 字母后面、右括号后面、或者前一个已经是下标的数字后面
                if prev_char.isalpha() or prev_char == ')':
                    use_sub_font = True
                elif is_prev_digit and is_prev_subscript:
                    use_sub_font = True

            is_prev_digit = True
            is_prev_subscript = use_sub_font
        else:
            is_prev_digit = False
            is_prev_subscript = False

        # 选择字体
        font = sub_font if use_sub_font else main_font

        # 计算 Y 坐标：下标从 Y + 偏移量开始，主字体从 Y 开始
        draw_y = y + subscript_offset_y if use_sub_font else y

        try:
            char_surf = font.render(char, True, color)
            surface.blit(char_surf, (current_x, draw_y))

            # 步进距离
            step = char_surf.get_width()
            # 细微调整：下标字符可以更紧凑一些
            if use_sub_font:
                step -= 1

            current_x += step

        except Exception as e:
            logging.error(f"渲染字符 '{char}' 失败: {e}")
            continue

    return current_x - x


def wrap_text(font, text, max_width):
    """将文本根据最大宽度进行换行"""
    if not text:
        return []

    lines = []
    current_line = ""
    # 使用空格分割，对于中文或其他连续文本，可以按字符处理
    segments = text.split(' ') if ' ' in text else list(text)

    for segment in segments:
        if current_line:
            # 尝试在当前行追加段落（包括一个空格）
            test_line = current_line + (" " if ' ' in text else "") + segment
        else:
            test_line = segment

        text_width = font.size(test_line)[0]

        if text_width <= max_width:
            current_line = test_line
        else:
            # 如果整段/整词就超宽，需要按字符分割
            if not current_line and ' ' not in text:
                temp_line = ""
                for i, char in enumerate(segment):
                    if font.size(temp_line + char)[0] <= max_width:
                        temp_line += char
                    else:
                        lines.append(temp_line)
                        temp_line = char
                if temp_line:
                    current_line = temp_line
            elif not current_line and ' ' in text:
                # 当前行空，但新段落超宽（例如一个超长的URL）
                temp_line = ""
                for char in segment:
                    if font.size(temp_line + char)[0] <= max_width:
                        temp_line += char
                    else:
                        lines.append(temp_line)
                        temp_line = char
                if temp_line:
                    current_line = temp_line
            else:
                # 当前行已满，将当前行结束并开始新行
                lines.append(current_line.strip())
                current_line = segment

    if current_line.strip():
        lines.append(current_line.strip())

    return lines


def extract_links(text):
    """从文本中提取URL链接"""
    url_pattern = r'https?://[^\s|]+'
    links = []
    for match in re.finditer(url_pattern, text):
        links.append({
            'url': match.group(),
            'start': match.start(),
            'end': match.end()
        })
    return links


def draw_text_with_links(surface, font, text, x, y, color, link_color=PRIMARY_BLUE):
    """绘制包含可点击链接的文本"""
    links = extract_links(text)
    link_rects = []

    if not links:
        text_surf = font.render(text, True, color)
        surface.blit(text_surf, (x, y))
        return text_surf, []

    current_x = x
    current_y = y

    # 预先进行换行处理，确保链接不会被截断
    # 假设 surface 已经是 content_rect.width 大小的临时 surface
    max_line_width = surface.get_width() - 20
    full_text_lines = wrap_text(font, text, max_line_width)
    line_height = font.get_height()

    current_y = y

    for line_text in full_text_lines:
        current_x = x

        # 查找当前行文本与原始文本的对应关系
        line_start_in_full = text.find(line_text)

        # 简化处理：由于 wrap_text 已经处理了换行，我们只处理当前行内的文本和链接
        temp_line_pos_x = current_x

        # 修复：初始化当前行已处理文本的结束位置，解决 UnboundLocalError
        last_end_in_line = 0

        for link_info in links:
            # 检查链接是否完全或部分在当前行内
            link_start_in_full = link_info['start']
            link_end_in_full = link_info['end']
            line_end_in_full = line_start_in_full + len(line_text)

            # 链接在当前行之前或之后，跳过
            if link_end_in_full <= line_start_in_full or link_start_in_full >= line_end_in_full:
                continue

            # 计算链接在当前行中的起始和结束索引
            link_start_in_line = max(0, link_start_in_full - line_start_in_full)
            link_end_in_line = min(len(line_text), link_end_in_full - line_start_in_full)

            # 绘制链接前的文本
            before_text = line_text[last_end_in_line:link_start_in_line]
            if before_text:
                before_surf = font.render(before_text, True, color)
                surface.blit(before_surf, (temp_line_pos_x, current_y))
                temp_line_pos_x += before_surf.get_width()

            # 绘制链接文本
            link_text = line_text[link_start_in_line:link_end_in_line]
            link_surf = font.render(link_text, True, link_color)

            pygame.draw.line(surface, link_color,
                             (temp_line_pos_x, current_y + link_surf.get_height() - 1),
                             (temp_line_pos_x + link_surf.get_width(), current_y + link_surf.get_height() - 1), 2)

            surface.blit(link_surf, (temp_line_pos_x, current_y))

            # 记录可点击矩形
            clickable_rect = pygame.Rect(temp_line_pos_x, current_y, link_surf.get_width(), link_surf.get_height())
            link_rects.append({
                'rect': clickable_rect,
                'url': link_info['url']
            })

            temp_line_pos_x += link_surf.get_width()
            last_end_in_line = link_end_in_line

        # 绘制链接后的文本
        if last_end_in_line < len(line_text):
            after_text = line_text[last_end_in_line:]
            after_surf = font.render(after_text, True, color)
            surface.blit(after_surf, (temp_line_pos_x, current_y))

        current_y += line_height

    return None, link_rects


# --- 【修改 1a】移除固定物质列表 ---
def load_substance_images(substances, image_dir="images"):
    substance_images = {}
    # 为了避免启动时报错，直接返回空字典，在需要时再尝试加载
    return {}

substance_images = load_substance_images([]) # 初始化为空

def load_background_image(image_path="images/1234.png"):
    """加载背景图片"""
    try:
        image = pygame.image.load(image_path).convert_alpha()
        logging.debug("成功加载背景图片")
        return image
    except pygame.error as e:
        logging.warning(f"无法加载背景图片 {image_path}: {e}")
        return None


background_image = load_background_image()


def query_ai_general_info(substances_str):
    """
    通过Kimi查询物质信息（单物质）或反应情况（多物质）。
    :param substances_str: 物质列表，用逗号或加号分隔，例如 "H2O", "Na, HCl"
    :return: 格式化的AI结果字符串
    """
    # 移除查询字符串中的等号，避免方程式提前解析
    substances_str = substances_str.replace('=', '').strip()

    substances_list = [s.strip() for s in re.split(r'[+,，]', substances_str) if s.strip()]

    if not substances_list:
        return "ERROR***请输入有效的物质名称"

    substance1 = substances_list[0]
    substance2_list = substances_list[1:]
    substance2 = ', '.join(substance2_list) if substance2_list else ""

    try:
        client = OpenAI(
            api_key=KIMI_API_KEY,
            base_url=KIMI_BASE_URL,
        )

        if len(substances_list) == 1:
            # 单物质查询
            prompt = f"""请提供物质 {substance1} 的详细信息。

**分析要求：**
1. 详细介绍该物质的基本性质、结构特点和主要用途（不少于150字）。
2. 提供相关的学习资源和参考链接。

**回答格式必须严格按照以下三段式格式输出，并且每段内容之间必须使用三个星号（***）作为唯一分隔符：**

INFO***物质的详细介绍（不少于150字，需包含基本性质和结构特点）***参考链接：https://www.ranktracker.com/zh/seo/glossary/link-text/

**请确保：**
* 第一段必须是 **INFO**。
* 第二段是详细的介绍文本，不能包含 `***` 字符。
* 第三段必须以 **`参考链接：`** 开头，后面紧跟一个完整的、可访问的 URL 链接（例如：`https://zh.wikipedia.org/wiki/水`）。
"""

            system_content = "你是一个资深的化学专家和化学教育工作者，擅长用通俗易懂的语言解释复杂的化学概念，并提供准确的学习资源。请始终按照指定的格式回答，不要添加额外的解释。"

        else:
            # 多物质反应查询 (格式保持不变)
            reactants_formula = substance1 + ' + ' + '+'.join(substance2_list)
            prompt = f"""请详细分析化学反应 {reactants_formula} 的情况。


分析要求：
1. 判断这两种物质是否能发生化学反应
2. 分析反应的条件（温度、压力、催化剂等）
3. 分析反应的类型（置换反应、化合反应、分解反应、复分解反应等）
4. 说明反应的现象（颜色变化、气体产生、沉淀生成、放热等）
5. 提供相关的学习资源和参考链接（百度百科直接询问化学品（比如水））

回答格式必须严格按照以下格式：（中间记得要加上***）
如果能发生反应：YES***反应方程式***反应条件和现象***参考链接：https://zh.wikipedia.org/wiki/(反应后生成物质)***详细的反应机理和应用说明（500字以内）
如果不能发生反应：NO***不能反应的具体原因和化学原理（500字以内，需要说明为什么不能反应）
（实例：YES
*** 2 HCl(aq) + 2 Na(s) → 2 NaCl(aq) + H₂(g)↑
*** 常温常压即可，无需催化剂；钠熔成银白色小球并快速游动，发出“嘶嘶”声，溶液放热，伴随无色气泡（H₂）逸出，点燃可听到轻微爆鸣。
*** 参考链接：https://zh.wikipedia.org/wiki/氯化钠
*** 机理：Na失电子被氧化成Na⁺，H⁺得电子还原为H₂；实验室可用此法制少量纯净H₂，工业上因成本高已淘汰。）
请确保：
- 反应方程式必须正确和平衡
- 链接必须是真实的化学学习网站
- 现象描述要具体和专业
- 原因解释要基于化学原理"""

            system_content = "你是一个资深的化学专家和化学教育工作者，擅长用通俗易懂的语言解释复杂的化学概念。你需要：\n1. 准确判断化学反应的可能性\n2. 提供正确的化学方程式\n3. 解释反应条件和现象\n4. 提供有用的学习资源\n5. 帮助学生理解化学反应的原理\n请始终按照指定的格式回答，不要添加额外的解释。"

        completion = client.chat.completions.create(
            model=KIMI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.6,
        )

        result = completion.choices[0].message.content
        logging.debug(f"Kimi AI Response: {result}")

        # 使用新的分隔符 '***' 解析并重组
        result_lines = result.strip().split('***')

        is_info = len(substances_list) == 1 and 'INFO' in result_lines[0]
        is_reaction = len(substances_list) > 1 and ('YES' in result_lines[0] or 'NO' in result_lines[0])

        if is_info or is_reaction:
            return '***'.join([line.strip() for line in result_lines])

        return "ERROR***AI返回格式错误，请尝试不同的查询"

    except Exception as e:
        logging.error(f"Kimi查询错误: {e}", exc_info=True)
        return "ERROR***Kimi查询发生异常"


# --- 【核心修改点】AI 动态生成物质列表的函数，已加入用户列表约束和不可反应物质约束 ---
def query_ai_substance_list(context_substance=None):
    """
    调用 Kimi AI 生成物质列表。
    :param context_substance: 如果提供，生成与该物质反应的物质列表；否则生成中心物质列表。
    :return: 物质列表 (list of str)，或 None（如果失败）
    """
    try:
        client = OpenAI(
            api_key=KIMI_API_KEY,
            base_url=KIMI_BASE_URL,
        )

        system_content = "你是一个资深的化学专家，专门为初中/高一学生设计化学实验和教学内容。请仅输出物质的化学式，不要包含任何额外的文字、解释或编号。"

        if context_substance:
            # 生成反应物列表 - 包含能反应和不能反应的物质（4能反应 + 2不能反应）
            prompt = f"""请针对初中/高一化学阶段，提供6个物质的化学式，用于与中心物质 {context_substance} 进行反应模拟。

**要求：**
1. 在这6个物质中，必须包含4个能与 {context_substance} 发生化学反应的物质。
2. 在这6个物质中，必须包含2个**不能**与 {context_substance} 发生化学反应的物质（惰性气体除外，应选择常见的酸、碱、盐、氧化物等）。
3. 物质必须是常见的、且反应原理符合初中/高一教学大纲。
4. 每个物质的化学式之间使用英文逗号 `,` 分隔。
5. 严格输出6个化学式。
6. **所有物质必须从以下列表中选取。尽量选择与上次不同的组合：**
{ALLOWED_SUBSTANCES_LIST}

**格式示例：**
Na,H2O,FeCl3,AgNO3,SiO2,C
"""
        else:
            # 生成中心物质列表 (保持原样)
            prompt = f"""请针对初中/高一化学阶段，提供6个常见的、具有代表性的物质的化学式，作为实验的中心物质。

**要求：**
1. 物质必须是常见的，如酸、碱、盐、氧化物、单质。
2. 每个物质的化学式之间使用英文逗号 `,` 分隔。
3. 严格输出6个化学式。
4. **所有物质必须从以下列表中选取。尽量选择与上次不同的组合：**
{ALLOWED_SUBSTANCES_LIST}

**格式示例：**
HCl,NaOH,CuSO4,CaCO3,Fe,H2O
"""

        completion = client.chat.completions.create(
            model=KIMI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
        )

        result = completion.choices[0].message.content
        logging.debug(f"AI Substance List Response: {result}")

        # 解析化学式列表
        substance_list = [s.strip() for s in result.split(',') if s.strip()]

        if len(substance_list) == 6:
            # 随机打乱列表，将能反应和不能反应的物质混合
            random.shuffle(substance_list)
            return substance_list
        else:
            logging.error(f"AI返回的物质数量不符: {len(substance_list)}个，期待6个")
            return None

    except Exception as e:
        logging.error(f"Kimi查询物质列表错误: {e}")
        return None

# --- 【修改 1b】修改 HandDetector 类 (略) ---
class HandDetector:
    # (保持原样不变)
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.cap = cv2.VideoCapture(0)

    def get_hand_position(self, frame):
        """
        获取手的中心位置并校准到 Pygame 屏幕坐标。
        """
        frame = cv2.flip(frame, 1)  # 水平翻转以校正摄像头镜像（视觉镜像）
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        hand_pos = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                wrist = hand_landmarks.landmark[0]
                middle_finger = hand_landmarks.landmark[9]

                # 计算手部中心的归一化坐标
                avg_x_norm = (wrist.x + middle_finger.x) / 2
                avg_y_norm = (wrist.y + middle_finger.y) / 2

                # 使用直接映射到屏幕坐标
                x_cursor = int(avg_x_norm * WIDTH)
                y_cursor = int(avg_y_norm * HEIGHT)

                hand_pos = (x_cursor, y_cursor)

                # 绘制手部地标
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        return frame, hand_pos

    def detect_palm_open(self, frame):
        """检测手掌是否张开（用于清除操作）"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                fingers = 0
                if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
                    fingers += 1
                for tip, root in [(8, 7), (12, 11), (16, 15), (20, 19)]:
                    if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[root].y:
                        fingers += 1

                return fingers >= 4
        return False

    def detect_fist(self, frame):
        """检测是否握拳（用于确认选择）"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                fingers = 0
                # 检查拇指和其他四个手指是否弯曲
                is_thumb_curled = hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x if \
                hand_landmarks.landmark[4].x < hand_landmarks.landmark[0].x else hand_landmarks.landmark[4].x > \
                                                                                 hand_landmarks.landmark[
                                                                                     0].x  # 简化判断，确保不伸直
                if is_thumb_curled: fingers += 1
                for tip, root in [(8, 7), (12, 11), (16, 15), (20, 19)]:
                    if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[root].y:
                        fingers += 1

                return fingers <= 1  # 如果伸直的手指少于等于1，认为是握拳
        return False

    def detect_two_hands(self, frame):
        """检测是否同时存在两只手"""
        if frame is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) >= 2:
                return True
        return False


class InputBox:
    """用于手动输入文本的输入框类"""

    def __init__(self, x, y, w, h, text=''):
        self.rect = pygame.Rect(x, y, w, h)
        self.color = BLACK
        self.text = text
        self.font = font_medium
        self.active = False
        self.cursor_visible = True
        self.cursor_timer = pygame.time.get_ticks()

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = not self.active
            else:
                self.active = False
            self.color = PRIMARY_BLUE if self.active else BLACK

        if event.type == pygame.KEYDOWN:
            if self.active:
                self.cursor_timer = pygame.time.get_ticks()  # Reset cursor blink
                if event.key == pygame.K_RETURN:
                    pass  # Handled by screen logic
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    new_text = self.text + event.unicode
                    # Check if new text exceeds the box width
                    text_surf = self.font.render(new_text, True, BLACK)
                    if text_surf.get_width() < self.rect.width - 20:
                        self.text = new_text

    def update(self):
        # Cursor blink
        if self.active and pygame.time.get_ticks() - self.cursor_timer > 500:
            self.cursor_visible = not self.cursor_visible
            self.cursor_timer = pygame.time.get_ticks()

    def draw(self, surface):
        pygame.draw.rect(surface, WHITE, self.rect, border_radius=8)
        pygame.draw.rect(surface, self.color, self.rect, 3, border_radius=8)

        # Render text (vertically centered)
        # 限制显示的文本长度，确保光标可见
        display_text = self.text
        text_w = self.font.render(display_text, True, BLACK).get_width()
        while text_w > self.rect.width - 40 and len(display_text) > 0:
            display_text = display_text[1:]
            text_w = self.font.render(display_text, True, BLACK).get_width()

        txt_surface = self.font.render(display_text, True, BLACK)
        text_y = self.rect.y + (self.rect.height - txt_surface.get_height()) // 2
        surface.blit(txt_surface, (self.rect.x + 10, text_y))

        # Draw cursor
        if self.active and self.cursor_visible:
            cursor_x = self.rect.x + 10 + txt_surface.get_width()
            cursor_y = self.rect.y + (self.rect.height - self.font.get_height()) // 2
            cursor_h = self.font.get_height()
            pygame.draw.line(surface, BLACK, (cursor_x, cursor_y), (cursor_x, cursor_y + cursor_h), 2)


class SelectionBox:
    def __init__(self, x, y, width, height, substance, is_center=False):
        self.rect = pygame.Rect(x, y, width, height)
        self.substance = substance
        self.is_center = is_center
        self.is_selected = False
        self.is_hovering = False
        # 从全局字典获取图片
        self.image = substance_images.get(substance)
        self.has_image = self.image is not None

    def draw(self, surface):
        text_color = BLACK

        if self.is_selected:
            bg_color = SUCCESS_GREEN
            border_color = SUCCESS_GREEN
            border_width = 6
        elif self.is_hovering:
            bg_color = HOVER_YELLOW
            border_color = HOVER_YELLOW
            border_width = 4
        elif self.is_center:
            bg_color = BACKGROUND_DARK  # 保持 BACKGROUND_DARK
            border_color = PRIMARY_BLUE
            border_width = 4
        else:
            bg_color = WHITE
            border_color = BLACK
            border_width = 2

        pygame.draw.rect(surface, bg_color, self.rect, border_radius=8)
        pygame.draw.rect(surface, border_color, self.rect, border_width, border_radius=8)

        # ====== 渲染化学式 ======
        if self.has_image:
            # 1. 计算文本总宽度 (使用临时Surface计算，避免影响主Surface)
            temp_surf = pygame.Surface((self.rect.width, font_small.get_height() + font_tiny.get_height()))
            temp_surf.fill(WHITE)
            temp_surf.set_colorkey(WHITE)

            total_width = render_chemical_formula(temp_surf, self.substance,
                                                  0, 0,
                                                  font_small, font_tiny, text_color)

            # 2. 计算居中位置
            text_x = self.rect.centerx - (total_width // 2)
            # 3. 调整基线 Y 坐标
            text_y = self.rect.centery + 40 - (font_small.get_height() // 3)

            # 绘制图片
            image_x = self.rect.centerx - 50
            image_y = self.rect.centery - 40
            surface.blit(self.image, (image_x, image_y))

            # 最终渲染到屏幕
            render_chemical_formula(surface, self.substance,
                                    text_x, text_y,
                                    font_small, font_tiny, text_color)
        else:
            # 1. 计算文本总宽度
            temp_surf = pygame.Surface((self.rect.width, font_large.get_height() + font_small.get_height()))
            temp_surf.fill(WHITE)
            temp_surf.set_colorkey(WHITE)
            total_width = render_chemical_formula(temp_surf, self.substance,
                                                  0, 0,
                                                  font_large, font_small, text_color)

            # 2. 计算居中位置
            text_x = self.rect.centerx - (total_width // 2)
            # 3. 调整基线 Y 坐标 (使用 ascent 保持垂直居中稳定)
            # 如果是小按钮，使用 font_medium 居中
            if self.rect.height < 100:
                font_to_use = font_medium
                text_x = self.rect.centerx - (font_to_use.size(self.substance)[0] // 2)
                text_y = self.rect.centery - (font_to_use.get_ascent() // 2)
                text_surf = font_to_use.render(self.substance, True, text_color)
                surface.blit(text_surf, (text_x, text_y))
            else:
                text_y = self.rect.centery - (font_large.get_ascent() // 2)
                render_chemical_formula(surface, self.substance,
                                        text_x, text_y,
                                        font_large, font_small, text_color)
        # ======================================================================

    def contains_point(self, pos):
        return self.rect.collidepoint(pos)

    def set_hover(self, hovering):
        self.is_hovering = hovering


class GameState:
    def __init__(self, hand_detector):
        self.hand_detector = hand_detector
        self.state = "load_center_substances" # 【修改 2a】新增加载状态
        self.center_substance = None
        self.selected_substances = []
        self.reaction_info = None
        self.ai_query_thread = None
        self.is_querying = False
        self.hand_pos = None
        self.last_query_str = ""
        # 【新增 2b】用于存储 AI 生成的物质列表
        self.center_substances_list = None
        self.available_reactants_list = None


    def reset_selected(self):
        self.selected_substances.clear()

    def reset_to_select_center(self):
        self.state = "load_center_substances" # 【修改 2c】返回时重新加载中心物质
        self.center_substance = None
        self.selected_substances.clear()
        self.reaction_info = None
        self.last_query_str = ""
        self.center_substances_list = None
        self.available_reactants_list = None


class ChemistryLearner:
    def __init__(self):
        self.hand_detector = HandDetector()
        self.game_state = GameState(self.hand_detector)
        self.clock = pygame.time.Clock()
        self.running = True

    # 统一的摄像头绘制函数
    def draw_camera_feed(self, screen, ret, frame):
        """统一在右上角绘制摄像头画面"""
        cam_x, cam_y, frame_width, frame_height = 0, 0, 0, 0  # 默认值
        if ret and frame is not None:
            # 翻转摄像头画面使其不镜像
            frame = cv2.flip(frame, 1)
            frame_width = WIDTH // 4
            frame_height = HEIGHT // 4
            frame_small = cv2.resize(frame, (frame_width, frame_height))

            frame_surface = pygame.image.frombuffer(
                frame_small.tobytes(),
                (frame_width, frame_height),
                'BGR'
            )

            # 统一放置在右上角
            cam_x = WIDTH - frame_width - 20
            cam_y = 20
            screen.blit(frame_surface, (cam_x, cam_y))

            # 绘制摄像头框边界
            pygame.draw.rect(screen, PRIMARY_BLUE,
                             (cam_x, cam_y, frame_width, frame_height), 3)
        return cam_x, cam_y, frame_width, frame_height

    # --- 【新增 3】加载中心物质列表的界面和逻辑 ---
    def screen_load_center_substances(self):
        """
        加载中心物质列表的等待界面
        """
        self.game_state.is_querying = True
        self.game_state.center_substances_list = None # 清空旧列表

        def load_substances():
            sub_list = query_ai_substance_list(context_substance=None)
            self.game_state.center_substances_list = sub_list
            self.game_state.is_querying = False

        self.game_state.ai_query_thread = threading.Thread(target=load_substances)
        self.game_state.ai_query_thread.start()

        start_time = pygame.time.get_ticks()

        while self.running and self.game_state.state == "load_center_substances":
            self.clock.tick(30)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.running = False
                    return

            ret, frame = self.hand_detector.cap.read()
            # 绘制界面
            if background_image:
                screen.blit(background_image, (0, 0))
            else:
                screen.fill(BACKGROUND_LIGHT)

            # 标题
            title = font_large.render("AI 正在准备实验物质列表...", True, PRIMARY_BLUE)
            title_rect = title.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50))
            screen.blit(title, title_rect)

            # 动画提示
            dots = "." * ((pygame.time.get_ticks() - start_time) // 500 % 4)
            loading = font_medium.render(f"请稍候{dots}", True, BLACK)
            loading_rect = loading.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 50))
            screen.blit(loading, loading_rect)

            # 绘制摄像头
            self.draw_camera_feed(screen, ret, frame)

            pygame.display.flip()

            # 检查是否加载完成
            if not self.game_state.is_querying:
                if self.game_state.center_substances_list is None:
                    # 加载失败，提供默认列表或重试
                    self.game_state.center_substances_list = ['HCl', 'NaOH', 'CuSO4', 'Fe', 'O2', 'CO2']
                    logging.warning("AI加载失败，使用默认物质列表")
                self.game_state.state = "select_center"
                return

    def screen_select_center(self):
        """
        第二个界面：选择中心物质，使用 AI 生成的列表。
        """
        # 【修改 4a】使用 AI 生成的列表
        display_substances = self.game_state.center_substances_list
        if not display_substances:
            # 理论上不会发生，但在切换状态后仍需检查
            self.game_state.state = "load_center_substances"
            return

        boxes = []

        # 3x2 网格布局
        positions = [
            (100, 200),
            (400, 200),
            (700, 200),
            (100, 450),
            (400, 450),
            (700, 450),
        ]

        # 【修改 4b】动态加载 SelectionBox
        for i, substance in enumerate(display_substances):
            x, y = positions[i]
            # 每次选择前都尝试更新图片，避免 SelectionBox 构造函数使用旧的 substance_images
            if substance not in substance_images:
                try:
                    image_path = os.path.join("images", f"{substance}.png")
                    if os.path.exists(image_path):
                        image = pygame.image.load(image_path).convert_alpha()
                        substance_images[substance] = pygame.transform.scale(image, (100, 100))
                except pygame.error as e:
                    logging.warning(f"无法加载 {substance} 的图片: {e}")

            boxes.append(SelectionBox(x, y, 220, 180, substance))

        # 手动查询按钮
        manual_search_box = SelectionBox(WIDTH - 250, HEIGHT - 120, 200, 70, "手动查询")
        boxes.append(manual_search_box)

        selected = None
        ret = False
        frame = None
        fist_detected_count = 0

        while not selected and self.running and self.game_state.state == "select_center":
            self.clock.tick(30)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.running = False
                    return
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = event.pos
                    if manual_search_box.contains_point(mouse_pos):
                        self.game_state.state = "manual_search"
                        return

                    for box in boxes[:-1]:
                        if box.contains_point(mouse_pos):
                            selected = box.substance
                            box.is_selected = True
                            break

            ret, frame = self.hand_detector.cap.read()
            if ret:
                frame, hand_pos = self.hand_detector.get_hand_position(frame)
                self.game_state.hand_pos = hand_pos

                if hand_pos and self.hand_detector.detect_fist(frame):
                    fist_detected_count += 1
                    if fist_detected_count >= 2:
                        for box in boxes[:-1]:
                            if box.contains_point(hand_pos):
                                selected = box.substance
                                box.is_selected = True
                                fist_detected_count = 0
                                break
                        if manual_search_box.contains_point(hand_pos):
                            self.game_state.state = "manual_search"
                            return
                else:
                    fist_detected_count = 0

                if hand_pos:
                    for box in boxes:
                        if box.contains_point(hand_pos):
                            box.set_hover(True)
                        else:
                            box.set_hover(False)

            if background_image:
                screen.blit(background_image, (0, 0))
            else:
                screen.fill(BACKGROUND_LIGHT)

            # 标题和提示
            title = font_large.render("元素之手——AI化学实验室", True, BLACK)
            title_rect = title.get_rect(center=(WIDTH // 2, 50))
            screen.blit(title, title_rect)

            subtitle = font_medium.render("选择中心反应物质 或 进入手动查询", True, PRIMARY_BLUE)
            subtitle_rect = subtitle.get_rect(center=(WIDTH // 2, 110))
            screen.blit(subtitle, subtitle_rect)

            hint_text = font_small.render("操作提示: 移动光标至物质框，握拳（Fist）进行选择 | ESC 退出", True,
                                          BACKGROUND_DARK)
            screen.blit(hint_text, (50, HEIGHT - 50))

            # 绘制所有物质框
            for box in boxes:
                box.draw(screen)

            # 绘制光标
            if self.game_state.hand_pos:
                pygame.draw.circle(screen, CURSOR_COLOR, self.game_state.hand_pos, CURSOR_RADIUS)

            # 绘制摄像头
            self.draw_camera_feed(screen, ret, frame)

            pygame.display.flip()

        if selected:
            self.game_state.center_substance = selected
            self.game_state.state = "load_reactants" # 【修改 4c】跳转到加载反应物状态
            logging.debug(f"选择了中心物质: {selected}")

    # --- 【新增 5】加载反应物列表的界面和逻辑 ---
    def screen_load_reactants(self):
        """
        加载可反应物质列表的等待界面
        """
        if not self.game_state.center_substance:
            self.game_state.state = "load_center_substances" # 异常情况，返回起始状态
            return

        self.game_state.is_querying = True
        self.game_state.available_reactants_list = None # 清空旧列表

        def load_reactants():
            # 【修改 5a】调用 AI 生成反应物列表
            reactants_list = query_ai_substance_list(context_substance=self.game_state.center_substance)
            self.game_state.available_reactants_list = reactants_list
            self.game_state.is_querying = False

        self.game_state.ai_query_thread = threading.Thread(target=load_reactants)
        self.game_state.ai_query_thread.start()

        start_time = pygame.time.get_ticks()

        while self.running and self.game_state.state == "load_reactants":
            self.clock.tick(30)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.game_state.reset_to_select_center() # ESC 返回起始状态
                    return

            ret, frame = self.hand_detector.cap.read()

            # 绘制界面
            if background_image:
                screen.blit(background_image, (0, 0))
            else:
                screen.fill(BACKGROUND_LIGHT)

            # 标题
            title = font_large.render(f"AI 正在为 {self.game_state.center_substance} 匹配反应物...", True, PRIMARY_BLUE)
            title_rect = title.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50))
            screen.blit(title, title_rect)

            # 动画提示
            dots = "." * ((pygame.time.get_ticks() - start_time) // 500 % 4)
            loading = font_medium.render(f"请稍候{dots}", True, BLACK)
            loading_rect = loading.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 50))
            screen.blit(loading, loading_rect)

            # 绘制摄像头
            self.draw_camera_feed(screen, ret, frame)

            pygame.display.flip()

            # 检查是否加载完成
            if not self.game_state.is_querying:
                # 【修改 5b】加载失败，使用中心物质生成一个简单的默认列表 (例如水、金属、氧气)
                if self.game_state.available_reactants_list is None:
                    logging.warning("AI加载反应物列表失败，使用默认列表")
                    # 使用 4个可能反应 + 2个惰性/不反应物质 (例如CO2, N2)
                    default_reactants = ['H2O', 'Na', 'Fe', 'HCl', 'CO2', 'N2']
                    # 尝试从默认列表中排除中心物质，但保留6个
                    final_list = [r for r in default_reactants if r != self.game_state.center_substance]
                    # 简单填充，确保有 6 个
                    if len(final_list) < 6:
                        for r in default_reactants:
                            if r != self.game_state.center_substance and r not in final_list:
                                final_list.append(r)
                                if len(final_list) == 6: break
                    self.game_state.available_reactants_list = final_list[:6] # 确保只取 6 个
                    # 随机打乱
                    random.shuffle(self.game_state.available_reactants_list)

                self.game_state.state = "playing"
                return


    def screen_playing(self):
        """
        第三个界面：选择其他反应物，使用 AI 生成的列表。
        """
        # 【修改 6a】使用 AI 生成的反应物列表
        available = self.game_state.available_reactants_list
        if not available:
            self.game_state.state = "load_reactants" # 异常情况，返回加载状态
            return

        top_substances = available[:3]
        bottom_substances = available[3:6]

        box_width, box_height = 200, 160
        gap_x, gap_y = 50, 50

        center_box = SelectionBox(50, 150, box_width, box_height,
                                  self.game_state.center_substance, is_center=True)

        all_boxes = []

        # 动态加载 SelectionBox (并尝试加载图片)
        for sub_list in [top_substances, bottom_substances]:
            for i, sub in enumerate(sub_list):
                if sub not in substance_images:
                    try:
                        image_path = os.path.join("images", f"{sub}.png")
                        if os.path.exists(image_path):
                            image = pygame.image.load(image_path).convert_alpha()
                            substance_images[sub] = pygame.transform.scale(image, (100, 100))
                    except pygame.error as e:
                        logging.warning(f"无法加载 {sub} 的图片: {e}")

        # 第一列
        for i, sub in enumerate(top_substances):
            x = WIDTH // 2 - 200
            y = 150 + i * (box_height + gap_y)
            all_boxes.append(SelectionBox(x, y, box_width, box_height, sub))

        # 第二列
        for i, sub in enumerate(bottom_substances):
            x = WIDTH // 2 - 200 + box_width + gap_x
            y = 150 + i * (box_height + gap_y)
            all_boxes.append(SelectionBox(x, y, box_width, box_height, sub))

        info_text = f"移动手部选择，握拳确认"
        message = ""
        message_time = 0
        max_selections = 1 # 限制只能选择一个额外物质

        fist_detected_count = 0
        ret = False
        frame = None
        two_hands_history = deque(maxlen=10)

        while self.running and self.game_state.state == "playing":
            self.clock.tick(30)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.game_state.reset_to_select_center()  # ESC 返回第一个界面
                        return

                # 鼠标点击选中物质逻辑 (备用)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = event.pos
                    for box in all_boxes:
                        if box.contains_point(mouse_pos):
                            if box.substance not in self.game_state.selected_substances:
                                self.game_state.selected_substances.append(box.substance)
                                message = f"已选择: {box.substance}"
                                message_time = pygame.time.get_ticks()

                                if len(self.game_state.selected_substances) >= max_selections:
                                    self.game_state.state = "reaction_info"
                                    reactants_str = self.game_state.center_substance + ' + ' + self.game_state.selected_substances[0]
                                    self.query_and_show_info(reactants_str)
                                    return
                            break

            ret, frame = self.hand_detector.cap.read()
            if ret:
                frame, hand_pos = self.hand_detector.get_hand_position(frame)
                self.game_state.hand_pos = hand_pos

                if hand_pos and self.hand_detector.detect_fist(frame):
                    fist_detected_count += 1
                    if fist_detected_count >= 2:
                        for box in all_boxes:
                            if box.contains_point(hand_pos):
                                if box.substance not in self.game_state.selected_substances:
                                    self.game_state.selected_substances.append(box.substance)
                                    message = f"已选择: {box.substance}"
                                    message_time = pygame.time.get_ticks()
                                    fist_detected_count = 0

                                    if len(self.game_state.selected_substances) >= max_selections:
                                        self.game_state.state = "reaction_info"
                                        reactants_str = self.game_state.center_substance + ' + ' + self.game_state.selected_substances[0]
                                        self.query_and_show_info(reactants_str)
                                        return
                                break
                else:
                    fist_detected_count = 0

                if self.hand_detector.detect_palm_open(frame):
                    self.game_state.reset_selected()
                    message = "已清除选择!"
                    message_time = pygame.time.get_ticks()
                    for box in all_boxes:
                        box.set_hover(False)

                if hand_pos:
                    for box in all_boxes:
                        if box.contains_point(hand_pos):
                            box.set_hover(True)
                        else:
                            box.set_hover(False)

                two_hands = self.hand_detector.detect_two_hands(frame)
                two_hands_history.append(two_hands)

                if sum(two_hands_history) >= 5:
                    self.game_state.reset_to_select_center()
                    return

            if pygame.time.get_ticks() - message_time > 2000:
                message = ""

            if background_image:
                screen.blit(background_image, (0, 0))
            else:
                screen.fill(BACKGROUND_LIGHT)

            # 标题
            title = font_large.render("化学反应模拟 - 选择反应物", True, BLACK)
            screen.blit(title, (50, 20))

            # 绘制中心物质框
            center_box.draw(screen)
            center_label = font_medium.render("中心物质", True, PRIMARY_BLUE)
            screen.blit(center_label, (center_box.rect.x, center_box.rect.y - 40))

            # 绘制所有物质框
            for box in all_boxes:
                if box.substance in self.game_state.selected_substances:
                    box.is_selected = True
                else:
                    box.is_selected = False
                box.draw(screen)

            # 已选择物质/消息/操作提示 (放在左下方)
            selected_text = f"已选择 ({len(self.game_state.selected_substances)}/{max_selections}): {', '.join(self.game_state.selected_substances) if self.game_state.selected_substances else '无'}"
            text_surf = font_medium.render(selected_text, True, PRIMARY_BLUE)
            screen.blit(text_surf, (50, HEIGHT - 200))

            if message:
                msg_surf = font_medium.render(message, True, ACCENT_ORANGE)
                screen.blit(msg_surf, (50, HEIGHT - 150))
            else:
                if len(self.game_state.selected_substances) > 0:
                    quick_hint = font_small.render("已选择物质, 正在分析反应... | 张开手清除 | 两只手/ESC 返回选择中心", True, SUCCESS_GREEN)
                else:
                    quick_hint = font_small.render("选择一种物质进行反应，将立即查询", True, BACKGROUND_DARK)
                screen.blit(quick_hint, (50, HEIGHT - 150))

            hint = font_small.render(info_text, True, BLACK)
            screen.blit(hint, (50, HEIGHT - 100))

            # 绘制光标
            if self.game_state.hand_pos:
                pygame.draw.circle(screen, CURSOR_COLOR, self.game_state.hand_pos, CURSOR_RADIUS)

            # 绘制摄像头
            self.draw_camera_feed(screen, ret, frame)

            pygame.display.flip()

    def screen_manual_search(self):
        """
        手动搜索界面
        """
        input_box = InputBox(WIDTH // 2 - 400, HEIGHT // 2 - 50, 600, 50, 'Fe + H2SO4')
        confirm_button = SelectionBox(WIDTH // 2 + 250, HEIGHT // 2 - 50, 150, 50, "确认查询")
        back_button = SelectionBox(50, HEIGHT - 100, 150, 50, "返回 (ESC)")

        ret, frame = False, None

        while self.running and self.game_state.state == "manual_search":
            self.clock.tick(30)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.game_state.state = "select_center"
                        return
                    if event.key == pygame.K_RETURN:
                        if input_box.text:
                            self.query_and_show_info(input_box.text)
                            return

                input_box.handle_event(event)

                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = event.pos
                    if confirm_button.contains_point(mouse_pos) and input_box.text:
                        self.query_and_show_info(input_box.text)
                        return
                    elif back_button.contains_point(mouse_pos):
                        self.game_state.state = "select_center"
                        return

            # Hand Detection Logic
            ret, frame = self.hand_detector.cap.read()
            if ret:
                frame, hand_pos = self.hand_detector.get_hand_position(frame)
                self.game_state.hand_pos = hand_pos

                if hand_pos:
                    confirm_button.set_hover(confirm_button.contains_point(hand_pos))
                    back_button.set_hover(back_button.contains_point(hand_pos))
                else:
                    confirm_button.set_hover(False)
                    back_button.set_hover(False)

            # Drawing
            if background_image:
                screen.blit(background_image, (0, 0))
            else:
                screen.fill(BACKGROUND_LIGHT)

            title = font_large.render("物质信息/反应查询", True, BLACK)
            screen.blit(title, (50, 20))

            hint = font_medium.render("请输入物质或反应物（用 + 或 , 分隔）:", True, BLACK)
            screen.blit(hint, (WIDTH // 2 - 400, HEIGHT // 2 - 100))

            input_box.update()
            input_box.draw(screen)
            confirm_button.draw(screen)
            back_button.draw(screen)

            # 操作提示
            op_hint = font_small.render("键盘输入内容，鼠标点击按钮确认/返回", True, BACKGROUND_DARK)
            screen.blit(op_hint, (50, HEIGHT - 50))

            # 绘制光标
            if self.game_state.hand_pos:
                pygame.draw.circle(screen, CURSOR_COLOR, self.game_state.hand_pos, CURSOR_RADIUS)

            # 绘制摄像头
            self.draw_camera_feed(screen, ret, frame)

            pygame.display.flip()

    def query_and_show_info(self, query_str):
        """查询信息（物质或反应）并显示结果"""
        self.game_state.state = "reaction_info"
        self.game_state.is_querying = True
        self.game_state.last_query_str = query_str  # 保存查询字符串

        def query():
            result = query_ai_general_info(query_str)
            self.game_state.reaction_info = {
                'reactants': query_str,
                'ai_result': result
            }
            self.game_state.is_querying = False

        self.game_state.ai_query_thread = threading.Thread(target=query)
        self.game_state.ai_query_thread.start()

    def screen_reaction_info(self):
        """反应信息界面，兼容物质信息和反应分析"""
        start_time = pygame.time.get_ticks()
        max_wait_time = 15000

        two_hands_history = deque(maxlen=10)

        current_links = []
        scroll_offset = 0
        max_scroll = 0

        while self.running:
            self.clock.tick(30)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_SPACE:
                        # 返回逻辑：如果查询源自手动查询界面，返回手动查询；否则返回实验台
                        substances_in_query = [s.strip() for s in re.split(r'[+,，]', self.game_state.last_query_str) if
                                               s.strip()]

                        is_multi_substance_query = len(substances_in_query) > 1 and self.game_state.center_substance

                        if is_multi_substance_query:
                            self.game_state.state = "playing"  # 实验台查询结果返回实验台
                        else:
                            self.game_state.state = "manual_search"  # 否则返回手动查询界面

                        self.game_state.selected_substances.clear()
                        return

                    elif event.key == pygame.K_UP:
                        scroll_offset = min(scroll_offset + 50, 0)
                    elif event.key == pygame.K_DOWN:
                        scroll_offset = max(scroll_offset - 50, -max_scroll)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        mouse_pos = pygame.mouse.get_pos()
                        for link_info in current_links:
                            # 必须将链接的相对位置加上滚动偏移和内容框的起始位置
                            rect_on_screen = link_info['rect'].copy()
                            rect_on_screen.y = content_rect.y + link_info['rect'].y + scroll_offset

                            if rect_on_screen.collidepoint(mouse_pos):
                                try:
                                    webbrowser.open(link_info['url'])
                                    logging.debug(f"打开链接: {link_info['url']}")
                                except Exception as e:
                                    logging.error(f"打开链接失败: {e}")
                                break

            ret, frame = self.hand_detector.cap.read()
            if ret:
                frame, hand_pos = self.hand_detector.get_hand_position(frame)
                self.game_state.hand_pos = hand_pos
            else:
                frame = None
                hand_pos = None

            if frame is not None:
                two_hands = self.hand_detector.detect_two_hands(frame)
                two_hands_history.append(two_hands)

                if sum(two_hands_history) >= 5:
                    self.game_state.reset_to_select_center()
                    return
            else:
                two_hands_history.clear()

            # 绘制界面
            if background_image:
                screen.blit(background_image, (0, 0))
            else:
                screen.fill(BACKGROUND_LIGHT)

            # 标题
            title = font_large.render("分析报告", True, BLACK)
            screen.blit(title, (50, 20))

            # 创建文本显示区域
            content_rect = pygame.Rect(50, 100, WIDTH - 100, HEIGHT - 200)

            # 绘制内容背景
            pygame.draw.rect(screen, WHITE, content_rect, border_radius=10)
            pygame.draw.rect(screen, BACKGROUND_DARK, content_rect, 2, border_radius=10)

            if self.game_state.is_querying:
                # 查询中
                loading = font_large.render("AI 正在分析信息，请稍候...", True, PRIMARY_BLUE)
                screen.blit(loading, (WIDTH // 2 - 300, HEIGHT // 2 - 50))

                # 超时检查
                if pygame.time.get_ticks() - start_time > max_wait_time:
                    self.game_state.is_querying = False
                    self.game_state.reaction_info = {
                        'reactants': 'Unknown',
                        'ai_result': 'ERROR***查询超时，请检查网络或API配置'
                    }
            else:
                # 显示反应结果
                if self.game_state.reaction_info:
                    info = self.game_state.reaction_info
                    current_links = []

                    # 创建临时Surface用于绘制所有内容
                    temp_surface = pygame.Surface((content_rect.width, 5000))  # 扩大临时表面
                    temp_surface.fill(WHITE)
                    temp_surface.set_colorkey(WHITE)

                    y_offset = 10
                    max_display_width = content_rect.width - 40

                    result = info['ai_result']
                    lines = [line.strip() for line in result.split('***') if line.strip()]

                    # --- 统一的查询内容显示 ---
                    reactants_text = f"查询内容: {info['reactants']}"
                    query_type_font = font_large
                    if len(lines) > 0 and 'INFO' in lines[0]:
                        query_type_font = font_medium

                    # 仅在 temp_surface 上计算渲染宽度，不实际渲染
                    temp_text_surf = pygame.Surface((content_rect.width, 1))
                    temp_text_surf.set_colorkey(BLACK)
                    total_width = render_chemical_formula(temp_text_surf, reactants_text, 0, 0, query_type_font,
                                                          font_medium, BLACK)

                    # 渲染到主临时 Surface
                    render_chemical_formula(temp_surface, reactants_text,
                                            20,
                                            y_offset + (font_large.get_height() // 2 - font_medium.get_height() // 2),
                                            query_type_font, font_medium, PRIMARY_BLUE)
                    y_offset += 70

                    # --- 核心逻辑：区分 INFO 和 YES/NO ---
                    if 'INFO' in lines[0]:
                        # --- 单物质信息逻辑 ---
                        result_text = font_medium.render(f"▶ 报告类型: 物质信息报告", True, SUCCESS_GREEN)
                        temp_surface.blit(result_text, (20, y_offset))
                        y_offset += 50

                        # 详细信息
                        if len(lines) > 1 and lines[1].strip():
                            detail_label = font_medium.render("【详细信息】:", True, BLACK)
                            temp_surface.blit(detail_label, (20, y_offset))
                            y_offset += 40

                            # 确保内容能被换行正确处理
                            info_content = lines[1].replace('\n', ' ').replace('\r', '')
                            info_lines = wrap_text(font_small, info_content, max_display_width)
                            for line in info_lines:
                                info_surf = font_small.render(line, True, BLACK)
                                temp_surface.blit(info_surf, (30, y_offset))
                                y_offset += 35

                        # 参考链接
                        if len(lines) > 2 and lines[2].strip():
                            link_label = font_medium.render("【参考链接】:", True, BLACK)
                            temp_surface.blit(link_label, (20, y_offset))
                            y_offset += 40

                            link_text = lines[2]
                            _, links = draw_text_with_links(
                                temp_surface, font_small, link_text, 30, y_offset,
                                BLACK, PRIMARY_BLUE
                            )

                            for link in links:
                                # 链接矩形需要相对于 content_rect.y 的位置，因为 temp_surface 是从 y=0 开始的
                                current_links.append({
                                    'rect': link['rect'],
                                    'url': link['url']
                                })

                            y_offset += font_small.get_height() + 10

                    elif 'YES' in lines[0] or 'NO' in lines[0]:
                        # --- 反应分析逻辑 ---
                        if 'YES' in lines[0]:
                            result_text = font_medium.render("▶ 结论: ✓ 能发生化学反应", True, SUCCESS_GREEN)
                        else:
                            result_text = font_medium.render("▶ 结论: ✗ 不能发生化学反应", True, ERROR_RED)

                        temp_surface.blit(result_text, (20, y_offset))
                        y_offset += 50

                        # 反应方程式 (YES)
                        if 'YES' in lines[0] and len(lines) > 1 and lines[1].strip():
                            eq_label = font_medium.render("【反应方程式】:", True, BLACK)
                            temp_surface.blit(eq_label, (20, y_offset))
                            y_offset += 40

                            equation_lines = wrap_text(font_small, lines[1], max_display_width)
                            for line in equation_lines:
                                render_chemical_formula(temp_surface, line,
                                                        30, y_offset,
                                                        font_small, font_tiny, PRIMARY_BLUE)
                                y_offset += 35

                        # 不能反应的原因 (NO)
                        elif 'NO' in lines[0] and len(lines) > 1 and lines[1].strip():
                            reason_label = font_medium.render("【不能反应的原因】:", True, BLACK)
                            temp_surface.blit(reason_label, (20, y_offset))
                            y_offset += 40

                            reason_lines = wrap_text(font_small, lines[1], max_display_width)
                            for line in reason_lines:
                                reason_surf = font_small.render(line, True, BLACK)
                                temp_surface.blit(reason_surf, (30, y_offset))
                                y_offset += 35

                        # 反应条件和现象 (YES)
                        if 'YES' in lines[0] and len(lines) > 2 and lines[2].strip():
                            condition_label = font_medium.render("【条件与现象】:", True, BLACK)
                            temp_surface.blit(condition_label, (20, y_offset))
                            y_offset += 40

                            condition_lines = wrap_text(font_small, lines[2], max_display_width)
                            for line in condition_lines:
                                cond_surf = font_small.render(line, True, BLACK)
                                temp_surface.blit(cond_surf, (30, y_offset))
                                y_offset += 35

                        # 参考链接
                        if 'YES' in lines[0] and len(lines) > 3 and lines[3].strip():
                            link_label = font_medium.render("【参考链接】:", True, BLACK)
                            temp_surface.blit(link_label, (20, y_offset))
                            y_offset += 40

                            link_text = lines[3]
                            _, links = draw_text_with_links(
                                temp_surface, font_small, link_text, 30, y_offset,
                                BLACK, PRIMARY_BLUE
                            )

                            for link in links:
                                current_links.append({
                                    'rect': link['rect'],
                                    'url': link['url']
                                })

                            y_offset += font_small.get_height() + 10

                        # 详细说明 (YES)
                        if 'YES' in lines[0] and len(lines) > 4 and lines[4].strip():
                            detail_label = font_medium.render("【反应机理与应用】:", True, BLACK)
                            temp_surface.blit(detail_label, (20, y_offset))
                            y_offset += 40

                            detail_lines = wrap_text(font_small, lines[4], max_display_width)
                            for line in detail_lines:
                                detail_surf = font_small.render(line, True, BLACK)
                                temp_surface.blit(detail_surf, (30, y_offset))
                                y_offset += 35

                    else:
                        error_text = font_medium.render("❌ 查询失败或AI返回格式错误", True, ERROR_RED)
                        temp_surface.blit(error_text, (20, y_offset))
                        y_offset += 60
                        hint_text = font_small.render("请检查网络或输入的查询内容", True, BLACK)
                        temp_surface.blit(hint_text, (20, y_offset))

                    # 计算最大滚动距离
                    max_scroll = max(0, y_offset - content_rect.height)
                    # 限制滚动偏移量
                    scroll_offset = max(scroll_offset, -max_scroll)

                    # 绘制内容到屏幕
                    screen.blit(temp_surface,
                                (content_rect.x, content_rect.y + scroll_offset),
                                (0, 0, content_rect.width, content_rect.height))

                    # 绘制滚动条
                    if max_scroll > 0:
                        scrollbar_height = max(20, content_rect.height * content_rect.height / (y_offset + 50))
                        scroll_ratio = (-scroll_offset) / max_scroll if max_scroll > 0 else 0
                        scrollbar_y = content_rect.y + scroll_ratio * (content_rect.height - scrollbar_height)
                        pygame.draw.rect(screen, PRIMARY_BLUE,
                                         (WIDTH - 20, scrollbar_y, 10, scrollbar_height), border_radius=5)

                    # 返回提示
                    back_hint = font_small.render(
                        "SPACE/ESC 返回 | 两只手返回选择中心 | 点击链接打开 ", True,
                        BACKGROUND_DARK)
                    screen.blit(back_hint, (50, HEIGHT - 50))

            # 绘制光标
            if self.game_state.hand_pos:
                pygame.draw.circle(screen, CURSOR_COLOR, self.game_state.hand_pos, CURSOR_RADIUS)

            # 绘制摄像头
            self.draw_camera_feed(screen, ret, frame)

            pygame.display.flip()

    def run(self):
        """主游戏循环"""
        while self.running:
            # 【修改 7】新增加载状态
            if self.game_state.state == "load_center_substances":
                self.screen_load_center_substances()
            elif self.game_state.state == "select_center":
                self.screen_select_center()
            elif self.game_state.state == "load_reactants": # 【新增 8】加载反应物状态
                self.screen_load_reactants()
            elif self.game_state.state == "playing":
                self.screen_playing()
            elif self.game_state.state == "manual_search":
                self.screen_manual_search()
            elif self.game_state.state == "reaction_info":
                self.screen_reaction_info()

        pygame.quit()
        self.hand_detector.cap.release()
        cv2.destroyAllWindows()
        sys.exit()


# ========== 主程序入口 ==========
if __name__ == "__main__":
    game = ChemistryLearner()
    game.run()