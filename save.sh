#!/bin/bash

# 定义文件夹路径
DIR="/nvme/yyh/threestudio/load/shapes/save"

# 遍历文件夹中的 .obj 文件
for file in "$DIR"/*.obj; do
    # 提取文件名，替换下划线为空格
    filename=$(basename "$file" .obj)
    prompt=${filename//_/ }

    # 构建并执行命令
    python launch.py --config configs/shape-control.yaml --train --gpu 0 \
        system.prompt_processor.prompt="$prompt, highly detailed, four view of the same object, 3d asset" \
        system.mesh_fitting_geometry.shape_init="mesh:$file" \
        system.mesh_geometry.shape_init="mesh:$file" \
        system.background.random_aug=true

    # 暂停20秒
    sleep 20
done
