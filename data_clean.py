import os
import re
import chardet
from pathlib import Path
from tqdm import tqdm  # 进度条库，查看处理进度
import multiprocessing as mp

# 配置参数
INPUT_DIR = r"D:\hzh\py_work\llm_nlp\data\THUCNews\THUCNews\体育"
OUTPUT_DIR = r"D:\hzh\py_work\llm_nlp\data\THUCNews\THUCNews\体育_cleaned"
MIN_TEXT_LENGTH = 50  # 过滤过短无效文本的阈值
NUM_PROCESSES = mp.cpu_count() - 1  # 多进程数，留1核给系统


def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read(1024)
    result = chardet.detect(raw_data)
    return result['encoding']


def clean_text(text):
    if not text:
        return ""

    # 去除控制字符，保留有效文本内容
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # 统一换行/制表符为空格
    text = text.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')
    # 仅保留中文、英文、数字和常用标点
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：""''（）【】《》、·\s]', '', text)
    # 合并多个空格为一个
    text = re.sub(r'\s+', ' ', text).strip()

    # 清理重复标点
    text = re.sub(r'，+', '，', text)
    text = re.sub(r'。+', '。', text)
    text = re.sub(r'！+', '！', text)
    text = re.sub(r'？+', '？', text)

    # 去除首尾标点
    text = text.strip('，。！？；：""''（）【】《》、·')
    return text


def process_single_file(file_path, output_dir):
    """处理单个文件的清洗逻辑"""
    try:
        encoding = detect_encoding(file_path)
        if not encoding:
            encoding = 'utf-8'

        # 读取文件内容，忽略编码错误
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            content = f.read()

        cleaned_content = clean_text(content)

        # 过滤过短文本
        if len(cleaned_content) < MIN_TEXT_LENGTH:
            return False, f"文本过短: {file_path}"

        os.makedirs(output_dir, exist_ok=True)

        # 保存清洗后的文件（统一UTF-8编码）
        file_name = os.path.basename(file_path)
        output_path = os.path.join(output_dir, file_name)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)

        return True, f"处理成功: {file_path}"

    except Exception as e:
        return False, f"处理失败 {file_path}: {str(e)}"


def batch_process_files(input_dir, output_dir):
    """批量处理文件（多进程加速）"""
    file_paths = list(Path(input_dir).glob("*.txt"))
    total_files = len(file_paths)

    print(f"发现 {total_files} 个txt文件，开始清洗...")

    tasks = [(str(file_path), output_dir) for file_path in file_paths]

    success_count = 0
    fail_count = 0
    fail_files = []

    # 多进程处理大量文件
    with mp.Pool(processes=NUM_PROCESSES) as pool:
        results = list(tqdm(pool.starmap(process_single_file, tasks), total=total_files, desc="清洗进度"))

    # 统计处理结果
    for success, msg in results:
        if success:
            success_count += 1
        else:
            fail_count += 1
            fail_files.append(msg)

    print(f"\n处理完成！")
    print(f"成功: {success_count} 个文件")
    print(f"失败: {fail_count} 个文件")

    # 保存失败文件列表，便于后续排查
    if fail_files:
        with open(os.path.join(output_dir, "失败文件列表.txt"), 'w', encoding='utf-8') as f:
            f.write("\n".join(fail_files))
        print(f"失败文件列表已保存至: {os.path.join(output_dir, '失败文件列表.txt')}")


if __name__ == "__main__":
    batch_process_files(INPUT_DIR, OUTPUT_DIR)

    # 生成清洗报告
    input_file_count = len(list(Path(INPUT_DIR).glob("*.txt")))
    output_file_count = len(list(Path(OUTPUT_DIR).glob("*.txt")))

    print(f"\n=== 清洗报告 ===")
    print(f"原始文件总数: {input_file_count}")
    print(f"清洗后文件数: {output_file_count}")
    print(f"清洗率: {output_file_count / input_file_count * 100:.2f}%")