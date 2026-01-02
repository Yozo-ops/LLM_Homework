import os


def batch_merge_txt_files():
    # 定义源目录和目标目录
    source_dir = r"D:\hzh\py_work\llm_nlp\data\THUCNews\THUCNews\体育"
    target_dir = r"D:\hzh\py_work\llm_nlp\data"

    # 每组文件数量
    batch_size = 5000

    try:
        # 1. 获取目录下所有txt文件并排序（保证顺序稳定）
        txt_files = [f for f in os.listdir(source_dir) if f.endswith('.txt')]
        txt_files.sort()

        # 检查文件数量
        total_files = len(txt_files)
        if total_files == 0:
            print("错误：源目录下未找到任何txt文件！")
            return

        # 2. 计算分组数量
        total_batches = (total_files + batch_size - 1) // batch_size  # 向上取整
        print(f"共发现 {total_files} 个txt文件，将分为 {total_batches} 组进行合并（每组{batch_size}个）")

        # 3. 逐批处理文件
        for batch_num in range(total_batches):
            # 计算当前批次的文件索引范围
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, total_files)
            current_batch_files = txt_files[start_idx:end_idx]

            print(
                f"\n开始处理第 {batch_num + 1}/{total_batches} 组（文件索引：{start_idx}-{end_idx - 1}，共{len(current_batch_files)}个文件）")

            # 初始化当前批次的合并内容
            batch_content = ""
            processed_count = 0

            # 读取当前批次的所有文件
            for i, file_name in enumerate(current_batch_files):
                file_path = os.path.join(source_dir, file_name)

                try:
                    # 读取文件内容（UTF-8编码兼容中文）
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # 每个文件内容后添加分隔符，方便区分
                        batch_content += f"=== 文件：{file_name} ===\n"
                        batch_content += content + "\n" + "=" * 80 + "\n\n"

                    processed_count += 1

                    # 打印进度（每50个文件）
                    if (i + 1) % 50 == 0:
                        print(f"  已处理 {i + 1}/{len(current_batch_files)} 个文件")

                except Exception as e:
                    print(f"  读取文件 {file_name} 时出错：{e}，已跳过该文件")
                    continue

            # 4. 生成目标文件名并保存
            target_filename = f"sports_merged_{start_idx}-{end_idx - 1}.txt"
            target_filepath = os.path.join(target_dir, target_filename)

            # 写入合并内容
            with open(target_filepath, 'w', encoding='utf-8') as f:
                f.write(batch_content)

            # 计算文件大小
            file_size = os.path.getsize(target_filepath) / 1024 / 1024
            print(f"  第 {batch_num + 1} 组合并完成！")
            print(f"  文件保存路径：{target_filepath}")
            print(f"  文件大小：{file_size:.2f} MB")
            print(f"  成功处理 {processed_count}/{len(current_batch_files)} 个文件")

        # 5. 输出汇总信息
        print("\n" + "=" * 80)
        print(f"所有文件处理完成！")
        print(f"总计处理文件：{total_files} 个")
        print(f"生成合并文件：{total_batches} 个")
        print(f"文件保存位置：{target_dir}")

    except Exception as e:
        print(f"\n程序执行出错：{e}")


if __name__ == "__main__":
    batch_merge_txt_files()