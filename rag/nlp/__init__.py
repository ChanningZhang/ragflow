#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import logging
import random
import os
from collections import Counter
from typing import List, Optional, Dict

from rag.utils import num_tokens_from_string
from . import rag_tokenizer
import re
import copy
import roman_numbers as r
from word2number import w2n
from cn2an import cn2an
from PIL import Image

import chardet

all_codecs = [
    'utf-8', 'gb2312', 'gbk', 'utf_16', 'ascii', 'big5', 'big5hkscs',
    'cp037', 'cp273', 'cp424', 'cp437',
    'cp500', 'cp720', 'cp737', 'cp775', 'cp850', 'cp852', 'cp855', 'cp856', 'cp857',
    'cp858', 'cp860', 'cp861', 'cp862', 'cp863', 'cp864', 'cp865', 'cp866', 'cp869',
    'cp874', 'cp875', 'cp932', 'cp949', 'cp950', 'cp1006', 'cp1026', 'cp1125',
    'cp1140', 'cp1250', 'cp1251', 'cp1252', 'cp1253', 'cp1254', 'cp1255', 'cp1256',
    'cp1257', 'cp1258', 'euc_jp', 'euc_jis_2004', 'euc_jisx0213', 'euc_kr',
    'gb18030', 'hz', 'iso2022_jp', 'iso2022_jp_1', 'iso2022_jp_2',
    'iso2022_jp_2004', 'iso2022_jp_3', 'iso2022_jp_ext', 'iso2022_kr', 'latin_1',
    'iso8859_2', 'iso8859_3', 'iso8859_4', 'iso8859_5', 'iso8859_6', 'iso8859_7',
    'iso8859_8', 'iso8859_9', 'iso8859_10', 'iso8859_11', 'iso8859_13',
    'iso8859_14', 'iso8859_15', 'iso8859_16', 'johab', 'koi8_r', 'koi8_t', 'koi8_u',
    'kz1048', 'mac_cyrillic', 'mac_greek', 'mac_iceland', 'mac_latin2', 'mac_roman',
    'mac_turkish', 'ptcp154', 'shift_jis', 'shift_jis_2004', 'shift_jisx0213',
    'utf_32', 'utf_32_be', 'utf_32_le', 'utf_16_be', 'utf_16_le', 'utf_7', 'windows-1250', 'windows-1251',
    'windows-1252', 'windows-1253', 'windows-1254', 'windows-1255', 'windows-1256',
    'windows-1257', 'windows-1258', 'latin-2'
]


def find_codec(blob):
    detected = chardet.detect(blob[:1024])
    if detected['confidence'] > 0.5:
        if detected['encoding'] == "ascii":
            return "utf-8"

    for c in all_codecs:
        try:
            blob[:1024].decode(c)
            return c
        except Exception:
            pass
        try:
            blob.decode(c)
            return c
        except Exception:
            pass

    return "utf-8"


QUESTION_PATTERN = [
    r"第([零一二三四五六七八九十百0-9]+)问",
    r"第([零一二三四五六七八九十百0-9]+)条",
    r"[\(（]([零一二三四五六七八九十百]+)[\)）]",
    r"第([0-9]+)问",
    r"第([0-9]+)条",
    r"([0-9]{1,2})[\. 、]",
    r"([零一二三四五六七八九十百]+)[ 、]",
    r"[\(（]([0-9]{1,2})[\)）]",
    r"QUESTION (ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)",
    r"QUESTION (I+V?|VI*|XI|IX|X)",
    r"QUESTION ([0-9]+)",
]


def has_qbullet(reg, box, last_box, last_index, last_bull, bull_x0_list):
    section, last_section = box['text'], last_box['text']
    q_reg = r'(\w|\W)*?(?:？|\?|\n|$)+'
    full_reg = reg + q_reg
    has_bull = re.match(full_reg, section)
    index_str = None
    if has_bull:
        if 'x0' not in last_box:
            last_box['x0'] = box['x0']
        if 'top' not in last_box:
            last_box['top'] = box['top']
        if last_bull and box['x0'] - last_box['x0'] > 10:
            return None, last_index
        if not last_bull and box['x0'] >= last_box['x0'] and box['top'] - last_box['top'] < 20:
            return None, last_index
        avg_bull_x0 = 0
        if bull_x0_list:
            avg_bull_x0 = sum(bull_x0_list) / len(bull_x0_list)
        else:
            avg_bull_x0 = box['x0']
        if box['x0'] - avg_bull_x0 > 10:
            return None, last_index
        index_str = has_bull.group(1)
        index = index_int(index_str)
        if last_section[-1] == ':' or last_section[-1] == '：':
            return None, last_index
        if not last_index or index >= last_index:
            bull_x0_list.append(box['x0'])
            return has_bull, index
        if section[-1] == '?' or section[-1] == '？':
            bull_x0_list.append(box['x0'])
            return has_bull, index
        if box['layout_type'] == 'title':
            bull_x0_list.append(box['x0'])
            return has_bull, index
        pure_section = section.lstrip(re.match(reg, section).group()).lower()
        ask_reg = r'(what|when|where|how|why|which|who|whose|为什么|为啥|哪)'
        if re.match(ask_reg, pure_section):
            bull_x0_list.append(box['x0'])
            return has_bull, index
    return None, last_index


def index_int(index_str):
    res = -1
    try:
        res = int(index_str)
    except ValueError:
        try:
            res = w2n.word_to_num(index_str)
        except ValueError:
            try:
                res = cn2an(index_str)
            except ValueError:
                try:
                    res = r.number(index_str)
                except ValueError:
                    return -1
    return res


def qbullets_category(sections):
    global QUESTION_PATTERN
    hits = [0] * len(QUESTION_PATTERN)
    for i, pro in enumerate(QUESTION_PATTERN):
        for sec in sections:
            if re.match(pro, sec) and not not_bullet(sec):
                hits[i] += 1
                break
    maxium = 0
    res = -1
    for i, h in enumerate(hits):
        if h <= maxium:
            continue
        res = i
        maxium = h
    return res, QUESTION_PATTERN[res]


BULLET_PATTERN = [[
    r"第[零一二三四五六七八九十百0-9]+(分?编|部分)",
    r"第[零一二三四五六七八九十百0-9]+章",
    r"第[零一二三四五六七八九十百0-9]+节",
    r"第[零一二三四五六七八九十百0-9]+条",
    r"[\(（][零一二三四五六七八九十百]+[\)）]",
], [
    r"第[0-9]+章",
    r"第[0-9]+节",
    r"[0-9]{,2}[\. 、]",
    r"[0-9]{,2}\.[0-9]{,2}[^a-zA-Z/%~-]",
    r"[0-9]{,2}\.[0-9]{,2}\.[0-9]{,2}",
    r"[0-9]{,2}\.[0-9]{,2}\.[0-9]{,2}\.[0-9]{,2}",
], [
    r"第[零一二三四五六七八九十百0-9]+章",
    r"第[零一二三四五六七八九十百0-9]+节",
    r"[零一二三四五六七八九十百]+[ 、]",
    r"[\(（][零一二三四五六七八九十百]+[\)）]",
    r"[\(（][0-9]{,2}[\)）]",
], [
    r"PART (ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)",
    r"Chapter (I+V?|VI*|XI|IX|X)",
    r"Section [0-9]+",
    r"Article [0-9]+"
]
]


def random_choices(arr, k):
    k = min(len(arr), k)
    return random.choices(arr, k=k)


def not_bullet(line):
    patt = [
        r"0", r"[0-9]+ +[0-9~个只-]", r"[0-9]+\.{2,}"
    ]
    return any([re.match(r, line) for r in patt])


def bullets_category(sections):
    global BULLET_PATTERN
    hits = [0] * len(BULLET_PATTERN)
    for i, pro in enumerate(BULLET_PATTERN):
        for sec in sections:
            for p in pro:
                if re.match(p, sec) and not not_bullet(sec):
                    hits[i] += 1
                    break
    maxium = 0
    res = -1
    for i, h in enumerate(hits):
        if h <= maxium:
            continue
        res = i
        maxium = h
    return res


def is_english(texts):
    eng = 0
    if not texts:
        return False
    for t in texts:
        if re.match(r"[ `a-zA-Z.,':;/\"?<>!\(\)-]", t.strip()):
            eng += 1
    if eng / len(texts) > 0.8:
        return True
    return False


def is_chinese(text):
    if not text:
        return False
    chinese = 0
    for ch in text:
        if '\u4e00' <= ch <= '\u9fff':
            chinese += 1
    if chinese / len(text) > 0.2:
        return True
    return False


def tokenize(d, t, eng):
    d["content_with_weight"] = t
    t = re.sub(r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>", " ", t)
    d["content_ltks"] = rag_tokenizer.tokenize(t)
    d["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(d["content_ltks"])


def tokenize_chunks(chunks, doc, eng, pdf_parser=None):
    res = []
    # wrap up as es documents
    for ii, ck in enumerate(chunks):
        if len(ck.strip()) == 0:
            continue
        logging.debug("-- {}".format(ck))
        d = copy.deepcopy(doc)
        if pdf_parser:
            try:
                d["image"], poss = pdf_parser.crop(ck, need_position=True)
                add_positions(d, poss)
                ck = pdf_parser.remove_tag(ck)
            except NotImplementedError:
                pass
        else:
            add_positions(d, [[ii]*5])
        tokenize(d, ck, eng)
        res.append(d)
    return res

def tokenize_chunks_with_images(chunks, doc, eng, images):
    res = []
    # wrap up as es documents
    for ii, (ck, image) in enumerate(zip(chunks, images)):
        if len(ck.strip()) == 0:
            continue
        logging.debug("-- {}".format(ck))
        d = copy.deepcopy(doc)
        d["image"] = image
        add_positions(d, [[ii]*5])
        tokenize(d, ck, eng)
        res.append(d)
    return res

def tokenize_table(tbls, doc, eng, batch_size=10):
    res = []
    # add tables
    for (img, rows), poss in tbls:
        if not rows:
            continue
        if isinstance(rows, str):
            d = copy.deepcopy(doc)
            tokenize(d, rows, eng)
            d["content_with_weight"] = rows
            if img:
                d["image"] = img
                d["doc_type_kwd"] = "image"
            if poss:
                add_positions(d, poss)
            res.append(d)
            continue
        de = "; " if eng else "； "
        for i in range(0, len(rows), batch_size):
            d = copy.deepcopy(doc)
            r = de.join(rows[i:i + batch_size])
            tokenize(d, r, eng)
            if img:
                d["image"] = img
                d["doc_type_kwd"] = "image"
            add_positions(d, poss)
            res.append(d)
    return res


def add_positions(d, poss):
    if not poss:
        return
    page_num_int = []
    position_int = []
    top_int = []
    for pn, left, right, top, bottom in poss:
        page_num_int.append(int(pn + 1))
        top_int.append(int(top))
        position_int.append((int(pn + 1), int(left), int(right), int(top), int(bottom)))
    d["page_num_int"] = page_num_int
    d["position_int"] = position_int
    d["top_int"] = top_int


def remove_contents_table(sections, eng=False):
    i = 0
    while i < len(sections):
        def get(i):
            nonlocal sections
            return (sections[i] if isinstance(sections[i],
                                              type("")) else sections[i][0]).strip()

        if not re.match(r"(contents|目录|目次|table of contents|致谢|acknowledge)$",
                        re.sub(r"( | |\u3000)+", "", get(i).split("@@")[0], flags=re.IGNORECASE)):
            i += 1
            continue
        sections.pop(i)
        if i >= len(sections):
            break
        prefix = get(i)[:3] if not eng else " ".join(get(i).split()[:2])
        while not prefix:
            sections.pop(i)
            if i >= len(sections):
                break
            prefix = get(i)[:3] if not eng else " ".join(get(i).split()[:2])
        sections.pop(i)
        if i >= len(sections) or not prefix:
            break
        for j in range(i, min(i + 128, len(sections))):
            if not re.match(prefix, get(j)):
                continue
            for _ in range(i, j):
                sections.pop(i)
            break


def make_colon_as_title(sections):
    if not sections:
        return []
    if isinstance(sections[0], type("")):
        return sections
    i = 0
    while i < len(sections):
        txt, layout = sections[i]
        i += 1
        txt = txt.split("@")[0].strip()
        if not txt:
            continue
        if txt[-1] not in ":：":
            continue
        txt = txt[::-1]
        arr = re.split(r"([。？！!?;；]| \.)", txt)
        if len(arr) < 2 or len(arr[1]) < 32:
            continue
        sections.insert(i - 1, (arr[0][::-1], "title"))
        i += 1


def title_frequency(bull, sections):
    bullets_size = len(BULLET_PATTERN[bull])
    levels = [bullets_size + 1 for _ in range(len(sections))]
    if not sections or bull < 0:
        return bullets_size + 1, levels

    for i, (txt, layout) in enumerate(sections):
        for j, p in enumerate(BULLET_PATTERN[bull]):
            if re.match(p, txt.strip()) and not not_bullet(txt):
                levels[i] = j
                break
        else:
            if re.search(r"(title|head)", layout) and not not_title(txt.split("@")[0]):
                levels[i] = bullets_size
    most_level = bullets_size + 1
    for level, c in sorted(Counter(levels).items(), key=lambda x: x[1] * -1):
        if level <= bullets_size:
            most_level = level
            break
    return most_level, levels


def not_title(txt):
    if re.match(r"第[零一二三四五六七八九十百0-9]+条", txt):
        return False
    if len(txt.split()) > 12 or (txt.find(" ") < 0 and len(txt) >= 32):
        return True
    return re.search(r"[,;，。；！!]", txt)


def hierarchical_merge(bull, sections, depth):
    if not sections or bull < 0:
        return []
    if isinstance(sections[0], type("")):
        sections = [(s, "") for s in sections]
    sections = [(t, o) for t, o in sections if
                t and len(t.split("@")[0].strip()) > 1 and not re.match(r"[0-9]+$", t.split("@")[0].strip())]
    bullets_size = len(BULLET_PATTERN[bull])
    levels = [[] for _ in range(bullets_size + 2)]

    for i, (txt, layout) in enumerate(sections):
        for j, p in enumerate(BULLET_PATTERN[bull]):
            if re.match(p, txt.strip()):
                levels[j].append(i)
                break
        else:
            if re.search(r"(title|head)", layout) and not not_title(txt):
                levels[bullets_size].append(i)
            else:
                levels[bullets_size + 1].append(i)
    sections = [t for t, _ in sections]

    # for s in sections: print("--", s)

    def binary_search(arr, target):
        if not arr:
            return -1
        if target > arr[-1]:
            return len(arr) - 1
        if target < arr[0]:
            return -1
        s, e = 0, len(arr)
        while e - s > 1:
            i = (e + s) // 2
            if target > arr[i]:
                s = i
                continue
            elif target < arr[i]:
                e = i
                continue
            else:
                assert False
        return s

    cks = []
    readed = [False] * len(sections)
    levels = levels[::-1]
    for i, arr in enumerate(levels[:depth]):
        for j in arr:
            if readed[j]:
                continue
            readed[j] = True
            cks.append([j])
            if i + 1 == len(levels) - 1:
                continue
            for ii in range(i + 1, len(levels)):
                jj = binary_search(levels[ii], j)
                if jj < 0:
                    continue
                if levels[ii][jj] > cks[-1][-1]:
                    cks[-1].pop(-1)
                cks[-1].append(levels[ii][jj])
            for ii in cks[-1]:
                readed[ii] = True

    if not cks:
        return cks

    for i in range(len(cks)):
        cks[i] = [sections[j] for j in cks[i][::-1]]
        logging.debug("\n* ".join(cks[i]))

    res = [[]]
    num = [0]
    for ck in cks:
        if len(ck) == 1:
            n = num_tokens_from_string(re.sub(r"@@[0-9]+.*", "", ck[0]))
            if n + num[-1] < 218:
                res[-1].append(ck[0])
                num[-1] += n
                continue
            res.append(ck)
            num.append(n)
            continue
        res.append(ck)
        num.append(218)

    return res


def naive_merge(sections, chunk_token_num=128, delimiter="\n。；！？"):
    if not sections:
        return []
    if isinstance(sections[0], type("")):
        sections = [(s, "") for s in sections]
    cks = [""]
    tk_nums = [0]

    def add_chunk(t, pos):
        nonlocal cks, tk_nums, delimiter
        tnum = num_tokens_from_string(t)
        if not pos:
            pos = ""
        if tnum < 8:
            pos = ""
        # Ensure that the length of the merged chunk does not exceed chunk_token_num  
        if cks[-1] == "" or tk_nums[-1] > chunk_token_num:

            if t.find(pos) < 0:
                t += pos
            cks.append(t)
            tk_nums.append(tnum)
        else:
            if cks[-1].find(pos) < 0:
                t += pos
            cks[-1] += t
            tk_nums[-1] += tnum

    dels = get_delimiters(delimiter)
    for sec, pos in sections:
        splited_sec = re.split(r"(%s)" % dels, sec)
        for sub_sec in splited_sec:
            if re.match(f"^{dels}$", sub_sec):
                continue
            add_chunk(sub_sec, pos)

    return cks


def naive_merge_with_images(texts, images, chunk_token_num=128, delimiter="\n。；！？"):
    if not texts or len(texts) != len(images):
        return [], []
    # Enuser texts is str not tuple, if it is tuple, convert to str (get the first item)
    if isinstance(texts[0], tuple):
        texts = [t[0] for t in texts]
    cks = [""]
    result_images = [None]
    tk_nums = [0]

    def add_chunk(t, image, pos=""):
        nonlocal cks, result_images, tk_nums, delimiter
        tnum = num_tokens_from_string(t)
        if not pos:
            pos = ""
        if tnum < 8:
            pos = ""
        # Ensure that the length of the merged chunk does not exceed chunk_token_num
        if cks[-1] == "" or tk_nums[-1] > chunk_token_num:
            if t.find(pos) < 0:
                t += pos
            cks.append(t)
            result_images.append(image)
            tk_nums.append(tnum)
        else:
            if cks[-1].find(pos) < 0:
                t += pos
            cks[-1] += t
            if result_images[-1] is None:
                result_images[-1] = image
            else:
                result_images[-1] = concat_img(result_images[-1], image)
            tk_nums[-1] += tnum

    dels = get_delimiters(delimiter)
    for text, image in zip(texts, images):
        splited_sec = re.split(r"(%s)" % dels, text)
        for sub_sec in splited_sec:
            if re.match(f"^{dels}$", sub_sec):
                continue
            add_chunk(text, image)

    return cks, result_images


def naive_merge_with_monkeyocr_images(texts, image_lists, monkeyocr_parser=None, chunk_token_num=128, delimiter="\n。；！？"):
    """
    专门处理 MonkeyOCR 图片列表的合并函数，支持位置信息合并和基于content_list.json的段落分割
    """
    logging.info(f"naive_merge_with_monkeyocr_images called with {len(texts)} texts, {len(image_lists)} image_lists")
    
    if not texts or len(texts) != len(image_lists):
        logging.error(f"Mismatch: texts={len(texts)}, image_lists={len(image_lists)}")
        return [], []
    
    # 确保 texts 是字符串列表
    if isinstance(texts[0], tuple):
        texts = [t[0] for t in texts]
        logging.info("Converted tuple texts to string list")
    
    # 获取页面尺寸信息
    page_width, page_height = None, None
    if monkeyocr_parser:
        try:
            position_data = monkeyocr_parser.get_position_data()
            if position_data and position_data.get('middle_json'):
                middle_json_data = position_data['middle_json']
                pdf_info = middle_json_data.get("pdf_info", [])
                if pdf_info:
                    page_size = pdf_info[0].get("page_size")
                    if page_size and len(page_size) == 2:
                        page_width, page_height = page_size[0], page_size[1]
                        logging.info(f"Got page size from parser: {page_width}x{page_height}")
                    else:
                        logging.error(f"Invalid page_size from parser: {page_size}")
                        raise ValueError(f"Invalid page_size from parser: {page_size}")
                else:
                    logging.error("No pdf_info in middle_json")
                    raise ValueError("No pdf_info in middle_json, cannot determine page size")
            else:
                logging.error("No position data from parser")
                raise ValueError("No position data from parser, cannot determine page size")
        except Exception as e:
            logging.error(f"Failed to get page size from parser: {e}")
            raise ValueError(f"Failed to get page size from parser: {e}")
    else:
        logging.error("No monkeyocr_parser provided")
        raise ValueError("No monkeyocr_parser provided, cannot determine page size")
    
    if not page_width or not page_height or page_width <= 0 or page_height <= 0:
        logging.error(f"Invalid page dimensions: {page_width}x{page_height}")
        raise ValueError(f"Invalid page dimensions: {page_width}x{page_height}")
    
    # 尝试使用content_list.json进行基于段落的分割
    content_list = monkeyocr_parser.get_content_list() if monkeyocr_parser else None
    if content_list:
        logging.info(f"Using content_list.json with {len(content_list)} paragraphs for chunk splitting")
        try:
            return _naive_merge_with_content_list(texts, image_lists, content_list, monkeyocr_parser, chunk_token_num, page_width, page_height)
        except Exception as e:
            logging.warning(f"Content list based splitting failed: {e}, falling back to delimiter-based splitting")
            return _naive_merge_with_delimiter(texts, image_lists, monkeyocr_parser, chunk_token_num, delimiter, page_width, page_height)
    else:
        logging.warning("No content_list.json available, falling back to delimiter-based splitting")
        return _naive_merge_with_delimiter(texts, image_lists, monkeyocr_parser, chunk_token_num, delimiter, page_width, page_height)


def _naive_merge_with_content_list(texts, image_lists, content_list, monkeyocr_parser, chunk_token_num, page_width, page_height):
    """
    基于content_list.json的段落进行chunk分割
    """
    logging.info(f"Processing {len(content_list)} content paragraphs")
    
    # 记录输入的详细信息
    for i, (text, img_list) in enumerate(zip(texts, image_lists)):
        logging.info(f"Input section {i}: text_len={len(text)}, images={len(img_list)}")
        if img_list:
            for j, img in enumerate(img_list):
                if hasattr(img, 'size'):
                    logging.info(f"  Input image {j}: size={img.size}")
    
    cks = [""]
    result_images = [None]
    tk_nums = [0]
    chunk_image_lists = [[]]
    chunk_image_paths = [[]]  # 记录每个chunk的图片路径
    
    # 为每个content创建一个映射到texts和image_lists的关系
    # 这里简化处理，假设content_list的顺序与texts的顺序对应
    text_content = "\n".join(texts)
    all_images = []
    image_paths = []  # 记录图片路径，用于位置匹配
    
    for img_list in image_lists:
        all_images.extend(img_list)
    
    logging.info(f"Total images available: {len(all_images)}")
    logging.info(f"Content list items: {len(content_list)}")
    for i, item in enumerate(content_list):
        logging.info(f"Content item {i}: type={item.get('type')}, text_len={len(item.get('text', ''))}")
    
    current_image_index = 0
    
    def add_chunk_from_content(content_text, content_images, content_image_paths=None):
        nonlocal cks, result_images, tk_nums, chunk_image_lists, chunk_image_paths
        if content_image_paths is None:
            content_image_paths = []
        # 过滤掉空文本
        if not content_text.strip():
            logging.info(f"Skipping empty content paragraph")
            return
        tnum = num_tokens_from_string(content_text)
        logging.info(f"Processing content paragraph: text_len={len(content_text)}, tokens={tnum}, images={len(content_images)}")
        # 检查是否可以添加到当前chunk
        if cks[-1] == "":
            # 第一个chunk，直接添加
            cks[-1] = content_text
            tk_nums[-1] = tnum
            chunk_image_lists[-1].extend(content_images)
            chunk_image_paths[-1].extend(content_image_paths)
            logging.info(f"Added to first chunk: {tnum} tokens, {len(content_images)} images")
        elif tk_nums[-1] + tnum <= chunk_token_num:
            # 可以添加到当前chunk
            cks[-1] += "\n" + content_text
            tk_nums[-1] += tnum
            chunk_image_lists[-1].extend(content_images)
            chunk_image_paths[-1].extend(content_image_paths)
            logging.info(f"Added to current chunk: total {tk_nums[-1]} tokens, {len(chunk_image_lists[-1])} images")
        else:
            # 需要新开一个chunk
            cks.append(content_text)
            result_images.append(None)  # 临时设为None，稍后处理
            tk_nums.append(tnum)
            chunk_image_lists.append(content_images[:])
            chunk_image_paths.append(content_image_paths[:])
            logging.info(f"Created new chunk {len(cks)-1}: {tnum} tokens, {len(content_images)} images")
    
    # 处理每个content段落
    for i, content_item in enumerate(content_list):
        if content_item.get('type') == 'text':
            content_text = content_item.get('text', '').strip()
            if not content_text:
                continue
                
            # 简化处理：假设图片按顺序分配
            # 实际应该根据content的位置和图片的位置进行精确匹配
            content_images = []
            content_image_paths = []
            if current_image_index < len(all_images):
                # 检查是否有图片引用
                if '![' in content_text and '](' in content_text:
                    # 有图片引用，分配一个图片
                    content_images = [all_images[current_image_index]]
                    content_image_paths = [all_images[current_image_index].path if hasattr(all_images[current_image_index], 'path') else None]
                    current_image_index += 1
                    logging.info(f"Allocated image {current_image_index-1} to text content {i}")
            
            add_chunk_from_content(content_text, content_images, content_image_paths)
        elif content_item.get('type') == 'image':
            # 处理图片类型的content
            img_path = content_item.get('img_path', '')
            if img_path and current_image_index < len(all_images):
                # 分配对应的图片
                content_images = [all_images[current_image_index]]
                content_image_paths = [img_path]
                image_paths.append(img_path)  # 记录图片路径
                current_image_index += 1
                logging.info(f"Allocated image {current_image_index-1} to image content {i}: {img_path}")
                # 为图片添加一个占位符文本
                add_chunk_from_content(f"[img: {os.path.basename(img_path)}]", content_images, content_image_paths)
            else:
                logging.warning(f"No image available for image content {i}: {img_path}")
        elif content_item.get('type') == 'table':
            # 处理表格类型的content
            table_body = content_item.get('table_body', '')
            if table_body:
                add_chunk_from_content(table_body, [], [])
    
    # 处理每个chunk的图片合并
    # 在合成前，将chunk_image_paths赋值到parser，供后续使用
    if hasattr(monkeyocr_parser, '__setattr__'):
        monkeyocr_parser._chunk_image_paths = chunk_image_paths

    logging.info(f"Processing {len(chunk_image_lists)} chunks for image combination")
    for i, images in enumerate(chunk_image_lists):
        logging.info(f"Processing chunk {i}: {len(images)} images")
        if not images:
            result_images[i] = None
            logging.info(f"Chunk {i}: No images")
        else:
            # 统一的图片处理逻辑
            logging.info(f"Chunk {i}: Processing {len(images)} images")
            for j, img in enumerate(images):
                if hasattr(img, 'size'):
                    logging.info(f"  Chunk {i} image {j}: size={img.size}")
            
            try:
                # 获取位置信息
                position_data = monkeyocr_parser.get_position_data()
                if position_data and position_data.get('middle_json'):
                    # 使用 MonkeyOCR 的位置信息处理图片
                    logging.info(f"MonkeyOCR: Processing {len(images)} images in chunk {i} with position info")
                    # 获取当前chunk对应的图片路径
                    chunk_image_paths = []
                    if hasattr(monkeyocr_parser, '_chunk_image_paths') and i < len(monkeyocr_parser._chunk_image_paths):
                        chunk_image_paths = monkeyocr_parser._chunk_image_paths[i]
                    result_images[i] = combine_images_with_monkeyocr_position(images, position_data, chunk_image_paths)
                    if result_images[i] and hasattr(result_images[i], 'size'):
                        logging.info(f"Chunk {i}: Combined image size={result_images[i].size}")
                else:
                    # 回退到垂直拼接
                    logging.info(f"MonkeyOCR: No position data available, using vertical combination for chunk {i}")
                    result_images[i] = concat_img_with_page_limit(images, max_width=page_width, max_height=page_height)
                    if result_images[i] and hasattr(result_images[i], 'size'):
                        logging.info(f"Chunk {i}: Vertically combined image size={result_images[i].size}")
            except Exception as e:
                logging.error(f"Error processing images for chunk {i}: {e}")
                result_images[i] = concat_img_with_page_limit(images, max_width=page_width, max_height=page_height)
                if result_images[i] and hasattr(result_images[i], 'size'):
                    logging.info(f"Chunk {i}: Fallback combined image size={result_images[i].size}")
    
    # 过滤掉空的chunk
    filtered_chunks = []
    filtered_images = []
    for i, (chunk, img) in enumerate(zip(cks, result_images)):
        if chunk.strip():
            filtered_chunks.append(chunk)
            filtered_images.append(img)
    
    logging.info(f"naive_merge_with_content_list completed: {len(filtered_chunks)} chunks (filtered from {len(cks)})")
    return filtered_chunks, filtered_images


def _naive_merge_with_delimiter(texts, image_lists, monkeyocr_parser, chunk_token_num, delimiter, page_width, page_height):
    """
    基于delimiter的传统chunk分割方式（回退方案）
    """
    logging.info("Using delimiter-based chunk splitting")
    
    # 记录输入的详细信息
    for i, (text, img_list) in enumerate(zip(texts, image_lists)):
        logging.info(f"Input section {i}: text_len={len(text)}, images={len(img_list)}")
        if img_list:
            for j, img in enumerate(img_list):
                if hasattr(img, 'size'):
                    logging.info(f"  Input image {j}: size={img.size}")
    
    cks = [""]
    result_images = [None]
    tk_nums = [0]
    chunk_image_lists = [[]]
    chunk_image_paths = [[]]  # 记录每个chunk的图片路径

    def add_chunk(t, image_list, pos=""):
        nonlocal cks, result_images, tk_nums, chunk_image_lists, delimiter
        
        # 过滤掉空文本
        if not t.strip():
            logging.info(f"Skipping empty text chunk")
            return
        
        tnum = num_tokens_from_string(t)
        if not pos:
            pos = ""
        if tnum < 8:
            pos = ""
        
        logging.info(f"add_chunk called: text_len={len(t)}, tokens={tnum}, images={len(image_list)}, current_chunk_tokens={tk_nums[-1] if tk_nums else 0}")
        
        # 确保合并后的chunk长度不超过chunk_token_num
        if cks[-1] == "" or tk_nums[-1] > chunk_token_num:
            if t.find(pos) < 0:
                t += pos
            cks.append(t)
            result_images.append(None)  # 临时设为None，稍后处理
            chunk_image_lists.append([])
            tk_nums.append(tnum)
            logging.info(f"Created new chunk {len(cks)-1}: {tnum} tokens")
        else:
            if cks[-1].find(pos) < 0:
                t += pos
            cks[-1] += t
            tk_nums[-1] += tnum
            logging.info(f"Merged to existing chunk {len(cks)-1}: total {tk_nums[-1]} tokens")
        
        # 收集图片到当前chunk
        if image_list:
            chunk_image_lists[-1].extend(image_list)
            logging.info(f"Added {len(image_list)} images to chunk {len(cks)-1}, total images in chunk: {len(chunk_image_lists[-1])}")
        else:
            logging.info(f"No images to add to chunk {len(cks)-1}")

    dels = get_delimiters(delimiter)
    logging.info(f"Text processing with delimiter: {dels}")
    
    for i, (text, image_list) in enumerate(zip(texts, image_lists)):
        logging.info(f"Processing text section {i}: {len(text)} chars, {len(image_list)} images")
        splited_sec = re.split(r"(%s)" % dels, text)
        logging.info(f"Section {i} split into {len(splited_sec)} parts")
        
        # 标记是否已经为当前section添加过图片
        section_images_added = False
        
        for j, sub_sec in enumerate(splited_sec):
            if re.match(f"^{dels}$", sub_sec):
                logging.info(f"Section {i} part {j}: delimiter, skipping")
                continue
            
            # 过滤掉空的sub_sec
            if not sub_sec.strip():
                logging.info(f"Section {i} part {j}: empty text, skipping")
                continue
            
            # 只有第一个有效的sub_sec才添加图片，避免重复
            if not section_images_added and image_list:
                logging.info(f"Section {i} part {j}: adding chunk with {len(sub_sec)} chars and {len(image_list)} images")
                add_chunk(sub_sec, image_list)
                section_images_added = True
            else:
                logging.info(f"Section {i} part {j}: adding chunk with {len(sub_sec)} chars, no images (already added or none available)")
                add_chunk(sub_sec, [])

    # 处理每个chunk的图片合并
    logging.info(f"Processing {len(chunk_image_lists)} chunks for image combination")
    for i, images in enumerate(chunk_image_lists):
        logging.info(f"Processing chunk {i}: {len(images)} images")
        if not images:
            result_images[i] = None
            logging.info(f"Chunk {i}: No images")
        else:
            # 不管单张还是多张图片，都使用统一的处理逻辑
            logging.info(f"Chunk {i}: Processing {len(images)} images")
            for j, img in enumerate(images):
                if hasattr(img, 'size'):
                    logging.info(f"  Chunk {i} image {j}: size={img.size}")
            
            if monkeyocr_parser:
                try:
                    # 获取位置信息
                    position_data = monkeyocr_parser.get_position_data()
                    logging.info(f"Chunk {i}: Got position data: {position_data is not None}")
                    if position_data and position_data.get('middle_json'):
                        # 使用 MonkeyOCR 的位置信息处理图片（单张和多张统一处理）
                        logging.info(f"MonkeyOCR: Processing {len(images)} images in chunk {i} with position info")
                        result_images[i] = combine_images_with_monkeyocr_position(images, position_data)
                        if result_images[i] and hasattr(result_images[i], 'size'):
                            logging.info(f"Chunk {i}: Combined image size={result_images[i].size}")
                    else:
                        # 回退到垂直拼接
                        logging.info(f"MonkeyOCR: No position data available, using vertical combination for chunk {i}")
                        result_images[i] = concat_img_with_page_limit(images, max_width=page_width, max_height=page_height)
                        if result_images[i] and hasattr(result_images[i], 'size'):
                            logging.info(f"Chunk {i}: Vertically combined image size={result_images[i].size}")
                except Exception as e:
                    logging.error(f"Error processing images for chunk {i}: {e}")
                    result_images[i] = concat_img_with_page_limit(images, max_width=page_width, max_height=page_height)
                    if result_images[i] and hasattr(result_images[i], 'size'):
                        logging.info(f"Chunk {i}: Fallback combined image size={result_images[i].size}")
            else:
                # 无MonkeyOCR解析器，使用页面尺寸限制的垂直拼接
                result_images[i] = concat_img_with_page_limit(images, max_width=page_width, max_height=page_height)
                if result_images[i] and hasattr(result_images[i], 'size'):
                    logging.info(f"Chunk {i}: Default combined image size={result_images[i].size}")
    
    # 过滤掉空的chunk
    filtered_chunks = []
    filtered_images = []
    for i, (chunk, img) in enumerate(zip(cks, result_images)):
        if chunk.strip():
            filtered_chunks.append(chunk)
            filtered_images.append(img)
    
    logging.info(f"naive_merge_with_delimiter completed: {len(filtered_chunks)} chunks (filtered from {len(cks)})")
    return filtered_chunks, filtered_images


def combine_images_with_monkeyocr_position(images: List[Image.Image], position_data: Dict, image_paths: List[str] = None) -> Optional[Image.Image]:
    """
    使用MonkeyOCR位置信息合并图片，并确保符合页面尺寸限制
    
    Args:
        images: 图片列表
        position_data: 位置数据
        image_paths: 图片路径列表，用于匹配位置信息
    """
    if not images:
        return None
    
    if len(images) == 1:
        logging.info("Single image, attempting position-based scaling")
    else:
        logging.info(f"Multiple images ({len(images)}), attempting position-based combination")
    
    if image_paths:
        logging.info(f"Image paths provided: {image_paths}")
    else:
        logging.info("No image paths provided, will use sequential matching")
    
    middle_json = position_data.get('middle_json')
    if not middle_json:
        logging.error("No middle_json in position_data")
        return concat_img_with_page_limit(images, max_width=595.3, max_height=841.9)
    
    # 获取页面尺寸信息
    pdf_info = middle_json.get("pdf_info", [])
    if not pdf_info:
        logging.error("No pdf_info found in middle.json")
        raise ValueError("No pdf_info found in middle.json, cannot determine page size")
    
    reference_page_size = pdf_info[0].get("page_size")
    if not reference_page_size or len(reference_page_size) != 2:
        logging.error(f"Invalid or missing page_size in pdf_info[0]: {reference_page_size}")
        raise ValueError(f"Invalid or missing page_size in pdf_info[0]: {reference_page_size}")
    
    logging.info(f"Using reference page size: {reference_page_size}")
    
    try:
        # 创建临时解析器来使用位置信息合并功能
        from deepdoc.parser.monkeyocr_parser import MonkeyOCRResultParser
        parser = MonkeyOCRResultParser()
        
        # 提取位置信息，去重并保持顺序
        all_positions = []
        seen_paths = set()
        
        # 遍历所有页面
        logging.info(f"Processing {len(pdf_info)} pages from middle.json")
        
        for page_info in pdf_info:
            page_idx = page_info.get("page_idx", 0)
            page_size = page_info.get("page_size")
            if not page_size or len(page_size) != 2:
                logging.error(f"Invalid or missing page_size in page {page_idx}: {page_size}")
                raise ValueError(f"Invalid or missing page_size in page {page_idx}: {page_size}")
            
            logging.info(f"Processing page {page_idx}, page_size={page_size}")
            
            # 从preproc_blocks中提取图片位置（优先）
            preproc_blocks = page_info.get("preproc_blocks", [])
            logging.info(f"Page {page_idx}: {len(preproc_blocks)} preproc_blocks")
            
            for block_i, block in enumerate(preproc_blocks):
                if block.get("type") == "image":
                    bbox = block.get("bbox", [])
                    logging.info(f"Page {page_idx} block {block_i}: image block with bbox={bbox}")
                    if len(bbox) == 4:
                        # 查找对应的图片路径
                        for block_item in block.get("blocks", []):
                            for line in block_item.get("lines", []):
                                for span in line.get("spans", []):
                                    if span.get("type") == "image":
                                        image_path = span.get("image_path", "")
                                        if image_path and image_path not in seen_paths:
                                            all_positions.append({
                                                "image_path": image_path,
                                                "bbox": bbox,
                                                "page_idx": page_idx,
                                                "page_size": page_size,
                                                "left": bbox[0],
                                                "top": bbox[1],
                                                "right": bbox[2],
                                                "bottom": bbox[3]
                                            })
                                            seen_paths.add(image_path)
                                            logging.info(f"Added position for {image_path}: bbox={bbox}, page={page_idx}, page_size={page_size}")
            
            # 从images数组中提取图片位置（备用，如果preproc_blocks中没有）
            images_array = page_info.get("images", [])
            logging.info(f"Page {page_idx}: {len(images_array)} images in images array")
            
            for img_i, img_info in enumerate(images_array):
                if img_info.get("type") == "image":
                    bbox = img_info.get("bbox", [])
                    logging.info(f"Page {page_idx} image {img_i}: bbox={bbox}")
                    if len(bbox) == 4:
                        for block_item in img_info.get("blocks", []):
                            for line in block_item.get("lines", []):
                                for span in line.get("spans", []):
                                    if span.get("type") == "image":
                                        image_path = span.get("image_path", "")
                                        if image_path and image_path not in seen_paths:
                                            all_positions.append({
                                                "image_path": image_path,
                                                "bbox": bbox,
                                                "page_idx": page_idx,
                                                "page_size": page_size,
                                                "left": bbox[0],
                                                "top": bbox[1],
                                                "right": bbox[2],
                                                "bottom": bbox[3]
                                            })
                                            seen_paths.add(image_path)
                                            logging.info(f"Added position for {image_path} (from images): bbox={bbox}, page={page_idx}, page_size={page_size}")
        
        if not all_positions:
            logging.warning("No position information found in middle.json")
            return concat_img_with_page_limit(images, max_width=reference_page_size[0], max_height=reference_page_size[1])
        
        # 按页面和位置排序
        all_positions.sort(key=lambda x: (x["page_idx"], x["top"], x["left"]))
        
        logging.info(f"Found {len(all_positions)} unique image positions")
        for i, pos in enumerate(all_positions):
            logging.info(f"Position {i+1}: {pos['image_path']} -> bbox={pos['bbox']}")
        
        # 根据图片路径匹配位置信息
        selected_positions = []
        if image_paths and len(image_paths) == len(images):
            # 使用路径匹配
            logging.info("Using path-based position matching")
            for img_path in image_paths:
                # 查找匹配的位置信息
                matched_position = None
                for pos in all_positions:
                    if pos['image_path'] == img_path or pos['image_path'].endswith(os.path.basename(img_path)):
                        matched_position = pos
                        logging.info(f"Matched {img_path} to position: {pos['bbox']}")
                        break
                
                if matched_position:
                    selected_positions.append(matched_position)
                else:
                    logging.warning(f"No position found for image: {img_path}")
                    # 如果没有找到匹配的位置，使用默认位置
                    selected_positions.append({
                        "image_path": img_path,
                        "bbox": [0, 0, 100, 100],
                        "page_idx": 0,
                        "page_size": reference_page_size,
                        "left": 0,
                        "top": 0,
                        "right": 100,
                        "bottom": 100
                    })
        else:
            # 回退到顺序匹配
            logging.info("Using sequential position matching")
            selected_positions = all_positions[:len(images)]
        
        if len(selected_positions) != len(images):
            logging.warning(f"Position count ({len(selected_positions)}) doesn't match image count ({len(images)})")
            # 即使位置信息不匹配，也要确保图片缩放到页面尺寸限制内
            return concat_img_with_page_limit(images, max_width=reference_page_size[0], max_height=reference_page_size[1])
        
        # 使用位置信息合并图片
        logging.info(f"Using position-based combination for {len(images)} images")
        logging.info(f"Selected positions: {selected_positions}")
        
        combined_image = parser._combine_images_by_position(images, selected_positions)
        if combined_image and hasattr(combined_image, 'size'):
            logging.info(f"Position-based combination result: size={combined_image.size}")
        else:
            logging.warning("Position-based combination returned None or invalid image")
            
        return combined_image
        
    except Exception as e:
        logging.error(f"Error in position-based image combination: {e}")
        logging.exception("Full traceback:")
        # 使用页面尺寸限制的回退方法
        fallback_result = concat_img_with_page_limit(images, max_width=reference_page_size[0], max_height=reference_page_size[1])
        if fallback_result and hasattr(fallback_result, 'size'):
            logging.info(f"Fallback combination result: size={fallback_result.size}")
        return fallback_result


def concat_img_multiple(images):
    """垂直拼接多张图片"""
    if not images:
        return None
    if len(images) == 1:
        return images[0]
    
    result = images[0]
    for img in images[1:]:
        result = concat_img(result, img)
    
    return result


def concat_img_with_page_limit(images, max_width, max_height):
    """
    在页面尺寸限制内合并图片
    如果垂直拼接超出限制，则尝试缩放或水平布局
    
    Args:
        images: 图片列表
        max_width: 最大宽度
        max_height: 最大高度
    """
    if not images:
        return None
    if len(images) == 1:
        return images[0]
    
    if max_width <= 0 or max_height <= 0:
        logging.error(f"Invalid page dimensions: {max_width}x{max_height}")
        raise ValueError(f"Invalid page dimensions: {max_width}x{max_height}")
    
    logging.info(f"concat_img_with_page_limit: {len(images)} images, max_size={max_width}x{max_height}")
    
    # 计算垂直拼接的总尺寸
    total_height = sum(img.height for img in images)
    max_img_width = max(img.width for img in images)
    
    logging.info(f"Vertical concat would be: {max_img_width}x{total_height}")
    
    # 如果垂直拼接在页面尺寸内，使用原始方法
    if max_img_width <= max_width and total_height <= max_height:
        logging.info("Using vertical concatenation (within page limits)")
        return concat_img_multiple(images)
    
    # 如果超出限制，尝试缩放所有图片
    scale_factor = min(max_width / max_img_width, max_height / total_height)
    if scale_factor < 1.0:
        logging.info(f"Scaling images by factor {scale_factor:.3f}")
        scaled_images = []
        for i, img in enumerate(images):
            new_width = int(img.width * scale_factor)
            new_height = int(img.height * scale_factor)
            scaled_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            scaled_images.append(scaled_img)
            logging.info(f"Scaled image {i}: {img.size} -> {scaled_img.size}")
        
        # 垂直拼接缩放后的图片
        result = scaled_images[0]
        for img in scaled_images[1:]:
            result = concat_img(result, img)
        
        logging.info(f"Final scaled concatenated image: {result.size}")
        return result
    
    # 如果仍然无法处理，返回第一张图片
    logging.warning("Unable to fit images within page limits, returning first image")
    return images[0]

def docx_question_level(p, bull=-1):
    txt = re.sub(r"\u3000", " ", p.text).strip()
    if p.style.name.startswith('Heading'):
        return int(p.style.name.split(' ')[-1]), txt
    else:
        if bull < 0:
            return 0, txt
        for j, title in enumerate(BULLET_PATTERN[bull]):
            if re.match(title, txt):
                return j + 1, txt
    return len(BULLET_PATTERN[bull]), txt


def concat_img(img1, img2):
    if img1 and not img2:
        return img1
    if not img1 and img2:
        return img2
    if not img1 and not img2:
        return None
    width1, height1 = img1.size
    width2, height2 = img2.size

    new_width = max(width1, width2)
    new_height = height1 + height2
    new_image = Image.new('RGB', (new_width, new_height))

    new_image.paste(img1, (0, 0))
    new_image.paste(img2, (0, height1))

    return new_image


def naive_merge_docx(sections, chunk_token_num=128, delimiter="\n。；！？"):
    if not sections:
        return [], []

    cks = [""]
    images = [None]
    tk_nums = [0]

    def add_chunk(t, image, pos=""):
        nonlocal cks, tk_nums, delimiter
        tnum = num_tokens_from_string(t)
        if tnum < 8:
            pos = ""
        if cks[-1] == "" or tk_nums[-1] > chunk_token_num:
            if t.find(pos) < 0:
                t += pos
            cks.append(t)
            images.append(image)
            tk_nums.append(tnum)
        else:
            if cks[-1].find(pos) < 0:
                t += pos
            cks[-1] += t
            images[-1] = concat_img(images[-1], image)
            tk_nums[-1] += tnum

    dels = get_delimiters(delimiter)
    for sec, image in sections:
        splited_sec = re.split(r"(%s)" % dels, sec)
        for sub_sec in splited_sec:
            if re.match(f"^{dels}$", sub_sec):
                continue
            add_chunk(sub_sec, image,"")

    return cks, images


def extract_between(text: str, start_tag: str, end_tag: str) -> list[str]:
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    return re.findall(pattern, text, flags=re.DOTALL)


def get_delimiters(delimiters: str):
    dels = []
    s = 0
    for m in re.finditer(r"`([^`]+)`", delimiters, re.I):
        f, t = m.span()
        dels.append(m.group(1))
        dels.extend(list(delimiters[s: f]))
        s = t
    if s < len(delimiters):
        dels.extend(list(delimiters[s:]))

    dels.sort(key=lambda x: -len(x))
    dels = [re.escape(d) for d in dels if d]
    dels = [d for d in dels if d]
    dels_pattern = "|".join(dels)

    return dels_pattern
