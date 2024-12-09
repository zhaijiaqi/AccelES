def remove_first_line(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    with open(file_path, 'w') as file:
        file.writelines(lines[1:])

# 示例文件路径
file_path = '/data/sub1/jqzhai/program/Topk-SpMV/data/matrices_for_testing/crawl_300d_2M/crawl-300d-2M.vec'

remove_first_line(file_path)
