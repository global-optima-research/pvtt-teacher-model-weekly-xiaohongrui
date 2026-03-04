# Teapot Sample (紫砂壶)

用于 Wan2.1-VACE zero-shot 测试的样本数据。

## 来源

- **Etsy listing**: 4436338649
- **产品**: Handmade Yixing Zisha Teapot

## 数据结构

```
teapot/
├── video.mp4              # 原始视频 (720×1280, 30fps, 15s)
├── video_frames/          # 提取的视频帧 (15fps, 225帧)
│   ├── 0001.jpg
│   ├── 0002.jpg
│   └── ...
├── reference_images/      # 独立产品图 (4张)
│   ├── ref_side.jpg       # 侧面 (image_1)
│   ├── ref_top.jpg        # 俯视 (image_3)
│   ├── ref_front.jpg      # 正面 (image_4)
│   └── ref_stand.jpg      # 木托上 (image_5)
├── image_*.jpg            # 原始商品图片 (10张)
└── metadata.json          # 原始元数据
```

## 视频信息

| 属性 | 值 |
|------|-----|
| 分辨率 | 720×1280 (竖版) |
| 帧率 | 30 fps |
| 时长 | 15s |
| 内容 | 手持展示茶壶，旋转展示 |

## 使用方式

### Zero-shot 测试

```bash
python baseline/wan2.1-vace/test_zero_shot.py \
    --video_path samples/teapot/video_frames/ \
    --reference_path samples/teapot/reference_images/ref_side.jpg \
    --prompt "handmade yixing zisha teapot, red clay, product display" \
    --test_case combined
```

### Step 1: 生成 Mask (GPU 服务器上运行)

```bash
# 使用 Grounded SAM 2 自动分割
cd samples/teapot
python segment_teapot.py --text_prompt "teapot."
```

输出：`masks/` 目录，225 个 PNG 文件

### Step 2: Zero-shot 测试

```bash
python baseline/wan2.1-vace/test_zero_shot.py \
    --video_path samples/teapot/video_frames/ \
    --mask_path samples/teapot/masks/ \
    --reference_path samples/teapot/reference_images/ref_side.jpg \
    --prompt "handmade yixing zisha teapot, red clay, product display" \
    --test_case combined
```

### 注意事项

1. 视频是手持展示，需要先用 Grounded SAM 2 分割茶壶生成 mask
2. 参考图是桌面静物照，与视频场景不同
3. 这正好可以测试模型的跨场景泛化能力
