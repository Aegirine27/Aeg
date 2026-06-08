"""
分析SAM生成的mask质量，找出异常/低质量的标注

评估指标:
1. 面孔率 (porosity) - 白色像素占比
2. 连通区域数量 - 过多可能噪声多，过少可能过度合并
3. 区域大小变异系数 - 衡量孔径分布均匀性
4. 与阈值法的偏差 - SAM vs 传统方法差异过大可能有问题
5. 小区域比例 - 过多微小孔可能是噪声
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from scipy import stats
from tqdm import tqdm

from src.utils import load_image, load_config
from src.preprocessing import ImagePreprocessor
from src.threshold_segment import ThresholdSegmenter


def analyze_mask(mask_path, image_path=None):
    """分析单个mask的质量指标"""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    # 二值化
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    h, w = binary.shape
    total_pixels = h * w

    # 1. 面孔率
    porosity = np.count_nonzero(binary) / total_pixels * 100

    # 2. 连通区域分析
    num_labels, labels, stats_info, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    # 去掉背景 (label 0)
    num_regions = num_labels - 1

    if num_regions == 0:
        return {
            'name': mask_path.stem.replace('_mask', ''),
            'porosity': 0.0,
            'num_regions': 0,
            'mean_area': 0.0,
            'median_area': 0.0,
            'area_std': 0.0,
            'cv_area': 0.0,
            'min_area': 0.0,
            'max_area': 0.0,
            'small_ratio': 0.0,  # < 50像素的区域占比
            'large_ratio': 0.0,  # > 10000像素的区域占比
            'solidity_mean': 0.0,
        }

    # 区域面积 (去掉背景)
    areas = stats_info[1:, cv2.CC_STAT_AREA]

    # 3. 面积统计
    mean_area = np.mean(areas)
    median_area = np.median(areas)
    area_std = np.std(areas)
    cv_area = area_std / mean_area if mean_area > 0 else 0  # 变异系数

    # 4. 小区域/大区域比例
    small_count = np.sum(areas < 50)
    large_count = np.sum(areas > 10000)
    small_ratio = small_count / num_regions * 100
    large_ratio = large_count / num_regions * 100

    # 5. 区域 solidity (面积/凸包面积)，衡量形状规则性
    solidities = []
    for i in range(1, num_labels):
        region_mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            area = cv2.contourArea(contours[0])
            hull = cv2.convexHull(contours[0])
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidities.append(area / hull_area)

    solidity_mean = np.mean(solidities) if solidities else 0

    return {
        'name': mask_path.stem.replace('_mask', ''),
        'porosity': porosity,
        'num_regions': num_regions,
        'mean_area': mean_area,
        'median_area': median_area,
        'area_std': area_std,
        'cv_area': cv_area,
        'min_area': np.min(areas),
        'max_area': np.max(areas),
        'small_ratio': small_ratio,
        'large_ratio': large_ratio,
        'solidity_mean': solidity_mean,
    }


def analyze_all_masks(labels_dir, images_dir, config_path='config.yaml'):
    """分析所有mask并找出异常值"""
    labels_dir = Path(labels_dir)
    images_dir = Path(images_dir)

    mask_files = sorted(labels_dir.glob('*_mask.png'))
    print(f"找到 {len(mask_files)} 个mask文件")
    print("=" * 80)

    results = []

    for mask_path in tqdm(mask_files, desc="分析mask"):
        name = mask_path.stem.replace('_mask', '')
        image_path = images_dir / f"{name}.jpg"

        info = analyze_mask(mask_path, image_path)
        if info:
            results.append(info)

    return results


def find_abnormal_masks(results):
    """基于统计规则找出异常mask"""
    if not results:
        return []

    print("\n" + "=" * 80)
    print("质量分析报告")
    print("=" * 80)

    # 提取各指标
    porosities = [r['porosity'] for r in results]
    num_regions_list = [r['num_regions'] for r in results]
    cv_areas = [r['cv_area'] for r in results]
    small_ratios = [r['small_ratio'] for r in results]

    # 计算统计量
    def stats_str(values):
        return f"均值={np.mean(values):.2f}, 中位数={np.median(values):.2f}, 标准差={np.std(values):.2f}"

    print(f"\n面孔率: {stats_str(porosities)}")
    print(f"区域数: {stats_str(num_regions_list)}")
    print(f"面积变异系数: {stats_str(cv_areas)}")
    print(f"小区域比例: {stats_str(small_ratios)}")

    # 找出异常值 (使用IQR方法)
    def find_outliers(values, threshold=1.5):
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        outliers = []
        for i, v in enumerate(values):
            if v < lower or v > upper:
                outliers.append((i, v, 'low' if v < lower else 'high'))
        return lower, upper, outliers

    print("\n" + "-" * 80)
    print("异常检测 (IQR方法)")
    print("-" * 80)

    abnormal_scores = {r['name']: 0 for r in results}
    abnormal_reasons = {r['name']: [] for r in results}

    # 1. 面孔率异常
    lower, upper, outliers = find_outliers(porosities)
    print(f"\n面孔率异常范围: < {lower:.2f}% 或 > {upper:.2f}%")
    for idx, val, direction in outliers:
        name = results[idx]['name']
        abnormal_scores[name] += 2
        abnormal_reasons[name].append(f"面孔率异常({val:.2f}%, {'偏低' if direction == 'low' else '偏高'})")
        print(f"  ⚠ {name}: {val:.2f}%")

    # 2. 区域数异常 (过多=噪声多)
    lower, upper, outliers = find_outliers(num_regions_list)
    print(f"\n区域数异常范围: < {lower:.0f} 或 > {upper:.0f}")
    for idx, val, direction in outliers:
        name = results[idx]['name']
        abnormal_scores[name] += 1
        if direction == 'high':
            abnormal_reasons[name].append(f"区域数过多({val}个，可能噪声多)")
        else:
            abnormal_reasons[name].append(f"区域数过少({val}个，可能过度合并)")
        print(f"  ⚠ {name}: {val}个区域")

    # 3. 小区域比例过高
    lower, upper, outliers = find_outliers(small_ratios)
    print(f"\n小区域比例异常范围: > {upper:.2f}%")
    for idx, val, direction in outliers:
        if direction == 'high':
            name = results[idx]['name']
            abnormal_scores[name] += 1
            abnormal_reasons[name].append(f"小区域比例过高({val:.1f}%，噪声多)")
            print(f"  ⚠ {name}: {val:.1f}%小区域")

    # 4. 面孔率为0或极低的
    print(f"\n面孔率极低 (< 1%):")
    for r in results:
        if r['porosity'] < 1.0:
            abnormal_scores[r['name']] += 3
            abnormal_reasons[r['name']].append(f"面孔率极低({r['porosity']:.2f}%)")
            print(f"  ⚠ {r['name']}: {r['porosity']:.2f}%")

    # 5. 面孔率过高的 (> 60%)
    print(f"\n面孔率过高 (> 60%):")
    for r in results:
        if r['porosity'] > 60:
            abnormal_scores[r['name']] += 2
            abnormal_reasons[r['name']].append(f"面孔率过高({r['porosity']:.2f}%)")
            print(f"  ⚠ {r['name']}: {r['porosity']:.2f}%")

    # 按异常分数排序
    sorted_abnormal = sorted(abnormal_scores.items(), key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 80)
    print("质量最差排名 (Top 10 建议人工检查)")
    print("=" * 80)

    top10 = []
    for name, score in sorted_abnormal[:10]:
        if score > 0:
            top10.append(name)
            r = next(x for x in results if x['name'] == name)
            print(f"\n🔴 {name} (异常分: {score})")
            print(f"   面孔率: {r['porosity']:.2f}%, 区域数: {r['num_regions']}, 小区域比例: {r['small_ratio']:.1f}%")
            for reason in abnormal_reasons[name]:
                print(f"   → {reason}")

    return top10, results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='分析SAM标注质量')
    parser.add_argument('--labels', '-l', default='data/labels', help='标注目录')
    parser.add_argument('--images', '-i', required=True, help='原始图像目录')
    parser.add_argument('--config', '-c', default='config.yaml', help='配置文件')
    args = parser.parse_args()

    results = analyze_all_masks(args.labels, args.images, args.config)
    top10, all_results = find_abnormal_masks(results)

    print("\n" + "=" * 80)
    print("总结: 以下图像建议人工重新标注")
    print("=" * 80)
    for name in top10:
        print(f"  - {name}")

    # 输出完整对比表
    print("\n" + "=" * 80)
    print("完整数据表")
    print("=" * 80)
    print(f"{'图像名':<25} {'面孔率%':>8} {'区域数':>8} {'平均面积':>10} {'小区域%':>8} {'变异系数':>8}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: x['porosity']):
        marker = " 🔴" if r['name'] in top10 else ""
        print(f"{r['name']:<25} {r['porosity']:>8.2f} {r['num_regions']:>8} {r['mean_area']:>10.1f} {r['small_ratio']:>8.1f} {r['cv_area']:>8.2f}{marker}")


if __name__ == '__main__':
    main()
