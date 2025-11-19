import os
from sklearn.model_selection import train_test_split

# 文件路径列表
file_paths = [
    'datasets/i2p/i2p_hate_prompts.txt',
    'datasets/i2p/i2p_harassment_prompts.txt',
    'datasets/i2p/i2p_violence_prompts.txt',
    'datasets/i2p/i2p_self-harm_prompts.txt',
    'datasets/i2p/i2p_sexual_prompts.txt',
    'datasets/i2p/i2p_shocking_prompts.txt',
    'datasets/i2p/i2p_illegal_activity_prompts.txt'
]

# 输出目录
output_train_dir = "datasets/train/"
output_test_dir = "datasets/test/"
# os.makedirs(output_dir, exist_ok=True)  # 创建输出文件夹

# 遍历每个文件并进行划分
for file_path in file_paths:
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        prompts = f.readlines()  # 按行读取所有prompts

    # 划分成训练集（80%）和测试集（20%）
    train_prompts, test_prompts = train_test_split(prompts, test_size=0.2, random_state=42)

    # 生成输出文件名
    base_name = os.path.basename(file_path).replace('.txt', '')
    train_file = os.path.join(output_train_dir, f"{base_name}_train.txt")
    test_file = os.path.join(output_test_dir, f"{base_name}_test.txt")

    # 保存训练集
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_prompts)

    # 保存测试集
    with open(test_file, 'w', encoding='utf-8') as f:
        f.writelines(test_prompts)

    print(f"Processed {file_path}:")
    print(f"  Train set saved to: {train_file}")
    print(f"  Test set saved to: {test_file}")

print("All files have been processed and split successfully!")
