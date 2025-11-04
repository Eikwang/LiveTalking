"""
优化的面部融合模块

集成了多种高级融合技术:
1. LAB颜色空间匹配
2. 光照补偿  
3. 泊松融合
4. 自适应羽化
5. 多尺度融合
"""

import cv2
import numpy as np
import time
from typing import Tuple, Optional, Union

class FastFaceBlending:
    """快速面部融合类 - 保持向后兼容"""
    
    def __init__(self):
        pass
    
    def fast_color_match_lab(self, source_face, target_face):
        """快速LAB颜色匹配"""
        try:
            source_lab = cv2.cvtColor(source_face, cv2.COLOR_BGR2LAB)
            target_lab = cv2.cvtColor(target_face, cv2.COLOR_BGR2LAB)
            
            source_mean = np.mean(source_lab, axis=(0, 1))
            target_mean = np.mean(target_lab, axis=(0, 1))
            
            matched_lab = source_lab.astype(np.float32)
            matched_lab += (target_mean - source_mean)
            matched_lab = np.clip(matched_lab, 0, 255)
            
            return cv2.cvtColor(matched_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        except:
            return source_face
    
    def fast_gaussian_blend(self, source_face, target_face, landmarks=None):
        """快速高斯融合"""
        try:
            mask = np.ones(source_face.shape[:2], dtype=np.float32)
            feather_size = min(source_face.shape[0], source_face.shape[1]) // 10
            
            if feather_size > 0:
                mask = cv2.erode(mask, np.ones((feather_size, feather_size), np.uint8))
                mask = cv2.GaussianBlur(mask, (feather_size*2+1, feather_size*2+1), feather_size/3)
            
            mask_3d = np.stack([mask] * 3, axis=2)
            return (source_face * mask_3d + target_face * (1 - mask_3d)).astype(np.uint8)
        except:
            return source_face
    
    def create_optimized_mask(self, face_shape, feather_amount=0.8):
        """创建优化掩码"""
        mask = np.ones(face_shape[:2], dtype=np.float32)
        feather_size = int(min(face_shape[0], face_shape[1]) * 0.1 * feather_amount)
        
        if feather_size > 0:
            mask = cv2.erode(mask, np.ones((feather_size, feather_size), np.uint8))
            mask = cv2.GaussianBlur(mask, (feather_size*2+1, feather_size*2+1), feather_size/3)
        
        return mask
    
    def blend_face_fast(self, source_face, target_face, landmarks=None, feather_amount=0.8):
        """快速面部融合（仅遮罩融合，不做调色）"""
        try:
            # 创建掩码并融合（移除颜色匹配，保留纯遮罩融合）
            mask = self.create_optimized_mask(source_face.shape, feather_amount)
            mask_3d = np.stack([mask] * 3, axis=2)
            result = (source_face * mask_3d + target_face * (1 - mask_3d)).astype(np.uint8)
            return result
        except Exception as e:
            print(f"快速融合失败: {e}")
            return target_face

class ImprovedFaceBlending:
    """改进的面部融合类 - 解决黑边问题"""

    def __init__(self):
        """初始化改进的面部融合器"""
        # 功能开关（默认禁用调色与泊松融合，采用纯遮罩融合）
        self.enable_poisson = False
        self.enable_color_matching = False
        self.enable_edge_smoothing = True
        self.enable_noise_reduction = True
        
        # 原有参数
        self.color_match_enabled = True
        self.lighting_compensation = True
        self.adaptive_feathering = True
        self.multi_scale_blending = True
        
        # 384模型专用参数
        self.feather_ratio_384 = 0.08  # 羽化范围比例
        self.min_feather_size = 8      # 最小羽化尺寸
        self.max_feather_size = 25     # 最大羽化尺寸
        
        # 颜色相关参数不再使用，保留占位避免外部引用报错
        self.saturation_factor = 0.0
        self.color_blend_ratio = 0.0
        self.max_saturation_boost = 1.0
        self.saturation_preservation = 1.0
        self.brightness_threshold = 0
        self.max_brightness_adjustment = 1.0
        
    def blend_face(self, pred_frame: np.ndarray, original_frame: np.ndarray,
                   bbox: Tuple[int, int, int, int],
                   feather_amount: float = 0.8) -> Optional[np.ndarray]:
        """
        面部融合（仅遮罩融合，不做任何调色/光照补偿/泊松融合）
        """
        try:
            y1, y2, x1, x2 = bbox
            # 验证输入
            if not self._validate_inputs(pred_frame, original_frame, bbox):
                return None
            # 调整预测面部尺寸
            target_width = x2 - x1
            target_height = y2 - y1
            resized_pred = cv2.resize(pred_frame, (target_width, target_height),
                                      interpolation=cv2.INTER_LANCZOS4)
            # 仅使用高斯/距离变换的遮罩融合，保持原视频颜色一致性
            result = self._advanced_gaussian_blend(resized_pred, original_frame, bbox, feather_amount)
            return result
        except Exception as e:
            print(f"融合失败: {e}")
            return None
    
    def _validate_inputs(self, pred_frame: np.ndarray, original_frame: np.ndarray, 
                        bbox: Tuple[int, int, int, int]) -> bool:
        """验证输入参数"""
        y1, y2, x1, x2 = bbox
        
        # 检查边界框
        if (y1 >= y2 or x1 >= x2 or 
            y1 < 0 or x1 < 0 or 
            y2 > original_frame.shape[0] or 
            x2 > original_frame.shape[1]):
            return False
        
        # 检查图像
        if pred_frame is None or original_frame is None:
            return False
        
        if len(pred_frame.shape) != 3 or len(original_frame.shape) != 3:
            return False
        
        return True
    
    def _match_colors_lab(self, pred_face: np.ndarray, original_frame: np.ndarray, 
                         bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """LAB颜色空间匹配 - 解决颜色不一致"""
        try:
            y1, y2, x1, x2 = bbox
            
            # 转换到LAB颜色空间
            pred_lab = cv2.cvtColor(pred_face, cv2.COLOR_BGR2LAB)
            
            # 获取周围区域用于颜色参考
            margin = min(20, (y2-y1)//4, (x2-x1)//4)
            
            # 扩展区域获取更多上下文
            ref_y1 = max(0, y1 - margin)
            ref_y2 = min(original_frame.shape[0], y2 + margin)
            ref_x1 = max(0, x1 - margin)
            ref_x2 = min(original_frame.shape[1], x2 + margin)
            
            reference_region = original_frame[ref_y1:ref_y2, ref_x1:ref_x2]
            ref_lab = cv2.cvtColor(reference_region, cv2.COLOR_BGR2LAB)
            
            # 计算统计信息
            pred_mean = np.mean(pred_lab, axis=(0, 1))
            pred_std = np.std(pred_lab, axis=(0, 1))
            
            ref_mean = np.mean(ref_lab, axis=(0, 1))
            ref_std = np.std(ref_lab, axis=(0, 1))
            
            # 颜色匹配 - 只调整A和B通道（色彩），保持L通道（亮度）相对稳定
            matched_lab = pred_lab.astype(np.float32)
            
            for i in range(3):
                if pred_std[i] > 0:
                    if i == 0:  # L通道 - 轻微调整
                        matched_lab[:, :, i] = (matched_lab[:, :, i] - pred_mean[i]) * 0.7 * (ref_std[i] / pred_std[i]) + ref_mean[i] * 0.3 + pred_mean[i] * 0.7
                    else:  # A, B通道 - 完全匹配
                        matched_lab[:, :, i] = (matched_lab[:, :, i] - pred_mean[i]) * (ref_std[i] / pred_std[i]) + ref_mean[i]
            
            # 限制范围
            matched_lab = np.clip(matched_lab, 0, 255)
            
            # 转换回BGR
            matched_bgr = cv2.cvtColor(matched_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
            
            return matched_bgr
            
        except Exception as e:
            print(f"颜色匹配失败: {e}")
            return pred_face
    
    def _gentle_edge_smoothing(self, adjusted_face: np.ndarray, original_face: np.ndarray, 
                              strength: float = 0.15) -> np.ndarray:
        """温和的边缘平滑处理"""
        try:
            h, w = adjusted_face.shape[:2]
            edge_width = max(3, min(8, h//20, w//20))  # 很小的边缘宽度
            
            # 创建边缘掩码
            mask = np.ones((h, w), dtype=np.float32)
            mask[:edge_width, :] = 0
            mask[-edge_width:, :] = 0
            mask[:, :edge_width] = 0
            mask[:, -edge_width:] = 0
            
            # 距离变换创建平滑过渡
            dist_transform = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
            edge_mask = np.clip(dist_transform / edge_width, 0, 1)
            
            # 应用非常温和的强度
            edge_mask = np.power(edge_mask, 0.8) * strength + (1 - strength)
            edge_mask_3d = np.stack([edge_mask] * 3, axis=2)
            
            # 温和融合
            result = adjusted_face * edge_mask_3d + original_face * (1 - edge_mask_3d)
            return result.astype(np.uint8)
            
        except Exception as e:
            print(f"温和边缘平滑失败: {e}")
            return adjusted_face

    
    def _compensate_lighting(self, pred_face: np.ndarray, original_frame: np.ndarray, 
                           bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """光照补偿"""
        try:
            y1, y2, x1, x2 = bbox
            
            # 计算周围区域的平均亮度
            margin = min(15, (y2-y1)//6, (x2-x1)//6)
            
            surrounding_regions = []
            
            # 上方
            if y1 >= margin:
                surrounding_regions.append(original_frame[y1-margin:y1, x1:x2])
            
            # 下方
            if y2 + margin < original_frame.shape[0]:
                surrounding_regions.append(original_frame[y2:y2+margin, x1:x2])
            
            # 左侧
            if x1 >= margin:
                surrounding_regions.append(original_frame[y1:y2, x1-margin:x1])
            
            # 右侧
            if x2 + margin < original_frame.shape[1]:
                surrounding_regions.append(original_frame[y1:y2, x2:x2+margin])
            
            if not surrounding_regions:
                return pred_face
            
            # 计算周围亮度
            surrounding_brightness = np.mean([np.mean(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)) 
                                            for region in surrounding_regions])
            
            # 计算面部亮度
            face_brightness = np.mean(cv2.cvtColor(pred_face, cv2.COLOR_BGR2GRAY))
            
            # 光照补偿
            if abs(surrounding_brightness - face_brightness) > 10:
                brightness_ratio = surrounding_brightness / (face_brightness + 1e-6)
                brightness_ratio = np.clip(brightness_ratio, 0.7, 1.3)  # 限制调整范围
                
                compensated = pred_face.astype(np.float32) * brightness_ratio
                compensated = np.clip(compensated, 0, 255).astype(np.uint8)
                
                return compensated
            
            return pred_face
            
        except Exception as e:
            print(f"光照补偿失败: {e}")
            return pred_face
    
    def _progressive_lab_adjustment(self, pred_face: np.ndarray, reference_region: np.ndarray, 
                                  diff_analysis: dict, strength: float = 0.3) -> np.ndarray:
        """渐进式LAB空间调整"""
        try:
            pred_lab = cv2.cvtColor(pred_face, cv2.COLOR_BGR2LAB).astype(np.float32)
            ref_lab = cv2.cvtColor(reference_region, cv2.COLOR_BGR2LAB)
            
            pred_mean = np.mean(pred_lab, axis=(0, 1))
            ref_mean = np.mean(ref_lab, axis=(0, 1))
            
            # 根据分析结果和强度参数调整各通道
            lightness_strength = min(strength, diff_analysis['lightness_diff'] / 100.0)
            chroma_strength = min(strength * 0.8, diff_analysis['chroma_diff'] / 80.0)
            
            # 非常保守的调整
            pred_lab[:, :, 0] += (ref_mean[0] - pred_mean[0]) * lightness_strength
            pred_lab[:, :, 1] += (ref_mean[1] - pred_mean[1]) * chroma_strength
            pred_lab[:, :, 2] += (ref_mean[2] - pred_mean[2]) * chroma_strength
            
            # 限制范围
            pred_lab[:, :, 0] = np.clip(pred_lab[:, :, 0], 0, 100)
            pred_lab[:, :, 1] = np.clip(pred_lab[:, :, 1], -128, 127)
            pred_lab[:, :, 2] = np.clip(pred_lab[:, :, 2], -128, 127)
            
            return cv2.cvtColor(pred_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
            
        except Exception as e:
            print(f"渐进式LAB调整失败: {e}")
            return pred_face
    
    def _progressive_hsv_adjustment(self, pred_face: np.ndarray, reference_region: np.ndarray, 
                                 diff_analysis: dict, strength: float = 0.2) -> np.ndarray:
        """渐进式HSV空间调整"""
        try:
            pred_hsv = cv2.cvtColor(pred_face, cv2.COLOR_BGR2HSV).astype(np.float32)
            ref_hsv = cv2.cvtColor(reference_region, cv2.COLOR_BGR2HSV)
            
            pred_mean = np.mean(pred_hsv, axis=(0, 1))
            ref_mean = np.mean(ref_hsv, axis=(0, 1))
            
            # 色调调整（考虑环形特性）
            h_diff = ref_mean[0] - pred_mean[0]
            if abs(h_diff) > 90:
                h_diff = h_diff - 180 if h_diff > 0 else h_diff + 180
            
            hue_strength = min(strength * 0.6, diff_analysis['hue_diff'] / 60.0)
            saturation_strength = min(strength * 0.8, diff_analysis['saturation_diff'] / 80.0)
            
            # 非常保守的调整
            pred_hsv[:, :, 0] = (pred_hsv[:, :, 0] + h_diff * hue_strength) % 180
            pred_hsv[:, :, 1] += (ref_mean[1] - pred_mean[1]) * saturation_strength
            
            # 限制范围
            pred_hsv[:, :, 1] = np.clip(pred_hsv[:, :, 1], 0, 255)
            pred_hsv[:, :, 2] = np.clip(pred_hsv[:, :, 2], 0, 255)
            
            return cv2.cvtColor(pred_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            
        except Exception as e:
            print(f"渐进式HSV调整失败: {e}")
            return pred_face

    
    def _blend_with_best_method(self, pred_face: np.ndarray, original_frame: np.ndarray,
                                bbox: Tuple[int, int, int, int], feather_amount: float) -> np.ndarray:
        """选择融合方法（已统一为遮罩融合）"""
        return self._advanced_gaussian_blend(pred_face, original_frame, bbox, feather_amount)
    
    def _poisson_blend(self, pred_face: np.ndarray, original_frame: np.ndarray, 
                      bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """泊松融合 - 最佳效果"""
        try:
            y1, y2, x1, x2 = bbox
            
            # 创建掩码
            mask = np.ones((y2-y1, x2-x1), dtype=np.uint8) * 255
            
            # 收缩掩码边缘以避免边界问题
            kernel_size = max(3, min(7, (y2-y1)//20, (x2-x1)//20))
            if kernel_size >= 3:
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                mask = cv2.erode(mask, kernel, iterations=1)
            
            # 中心点
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # 泊松融合
            result = cv2.seamlessClone(pred_face, original_frame, mask, center, cv2.NORMAL_CLONE)
            
            return result
            
        except Exception as e:
            print(f"泊松融合失败: {e}")
            return None
    
    def _advanced_gaussian_blend(self, pred_face: np.ndarray, original_frame: np.ndarray, 
                                bbox: Tuple[int, int, int, int], feather_amount: float) -> np.ndarray:
        """高级高斯融合 - 备用方案"""
        y1, y2, x1, x2 = bbox
        result = original_frame.copy()
        
        # 计算自适应羽化尺寸
        face_width = x2 - x1
        face_height = y2 - y1
        
        if self.adaptive_feathering:
            # 基于面部尺寸的自适应羽化
            feather_size = int(min(face_width, face_height) * self.feather_ratio_384 * feather_amount)
            feather_size = max(self.min_feather_size, min(self.max_feather_size, feather_size))
        else:
            feather_size = int(15 * feather_amount)
        
        # 创建距离变换掩码
        mask = np.ones((face_height, face_width), dtype=np.float32)
        
        if feather_size > 0:
            # 使用距离变换创建更平滑的掩码
            border_mask = np.zeros((face_height, face_width), dtype=np.uint8)
            border_mask[feather_size:-feather_size, feather_size:-feather_size] = 255
            
            # 距离变换
            dist_transform = cv2.distanceTransform(border_mask, cv2.DIST_L2, 5)
            
            # 归一化到0-1
            if np.max(dist_transform) > 0:
                mask = np.clip(dist_transform / feather_size, 0, 1)
            
            # 应用高斯平滑
            mask = cv2.GaussianBlur(mask, (feather_size*2+1, feather_size*2+1), feather_size/3)
        
        # 标准融合
        mask_3d = np.stack([mask] * 3, axis=2)
        blended_region = (pred_face * mask_3d + 
                        original_frame[y1:y2, x1:x2] * (1 - mask_3d))
        result[y1:y2, x1:x2] = blended_region.astype(np.uint8)
        
        return result

    
    def _match_colors_lab_balanced(self, pred_face: np.ndarray, original_frame: np.ndarray, 
                                 bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """自适应颜色匹配算法 - 根据色差程度选择最佳策略"""
        try:
            y1, y2, x1, x2 = bbox
            
            # 1. 分析颜色差异程度
            diff_analysis = self._analyze_color_difference_severity(pred_face, original_frame, bbox)
            
            total_score = diff_analysis['total_score']
            dominant_issue = diff_analysis['dominant_issue']
            
            print(f"🎨 色差分析: 总分={total_score:.1f}, 主要问题={dominant_issue}")
            
            # 2. 根据色差程度和类型选择策略
            if total_score < 15:  # 轻微色差 - 最小调整
                print("   策略: 最小调整")
                return self._minimal_color_adjustment(pred_face, original_frame, bbox, diff_analysis)
            elif total_score < 40:  # 中等色差 - 针对性调整
                print("   策略: 针对性调整")
                return self._targeted_color_adjustment(pred_face, original_frame, bbox, diff_analysis)
            else:  # 严重色差 - 强化调整
                print("   策略: 强化调整")
                return self._enhanced_color_adjustment(pred_face, original_frame, bbox, diff_analysis)
            
        except Exception as e:
            print(f"颜色匹配过程中出现错误: {e}")
            return pred_face
    
    def _analyze_color_difference_severity(self, pred_face: np.ndarray, original_frame: np.ndarray, 
                                          bbox: Tuple[int, int, int, int]) -> dict:
        """分析颜色差异严重程度 - 使用更科学的方法"""
        try:
            y1, y2, x1, x2 = bbox
            
            # 获取参考区域
            margin = min(20, (y2-y1)//4, (x2-x1)//4)
            ref_y1 = max(0, y1 - margin)
            ref_y2 = min(original_frame.shape[0], y2 + margin)
            ref_x1 = max(0, x1 - margin)
            ref_x2 = min(original_frame.shape[1], x2 + margin)
            
            reference_region = original_frame[ref_y1:ref_y2, ref_x1:ref_x2]
            
            # 多颜色空间分析
            pred_lab = cv2.cvtColor(pred_face, cv2.COLOR_BGR2LAB).astype(np.float32)
            ref_lab = cv2.cvtColor(reference_region, cv2.COLOR_BGR2LAB).astype(np.float32)
            
            pred_hsv = cv2.cvtColor(pred_face, cv2.COLOR_BGR2HSV).astype(np.float32)
            ref_hsv = cv2.cvtColor(reference_region, cv2.COLOR_BGR2HSV).astype(np.float32)
            
            # 计算各通道差异
            pred_lab_mean = np.mean(pred_lab, axis=(0, 1))
            ref_lab_mean = np.mean(ref_lab, axis=(0, 1))
            
            pred_hsv_mean = np.mean(pred_hsv, axis=(0, 1))
            ref_hsv_mean = np.mean(ref_hsv, axis=(0, 1))
            
            # 1. 亮度差异 (LAB L通道)
            lightness_diff = abs(pred_lab_mean[0] - ref_lab_mean[0])
            
            # 2. 色度差异 (LAB a,b通道)
            chroma_diff = np.sqrt((pred_lab_mean[1] - ref_lab_mean[1])**2 + 
                                (pred_lab_mean[2] - ref_lab_mean[2])**2)
            
            # 3. 色调差异 (HSV H通道，考虑环形特性)
            h_diff = abs(pred_hsv_mean[0] - ref_hsv_mean[0])
            h_diff = min(h_diff, 180 - h_diff)  # 处理色调环形特性
            
            # 4. 饱和度差异 (HSV S通道)
            saturation_diff = abs(pred_hsv_mean[1] - ref_hsv_mean[1])
            
            # 5. 计算类似Delta E的综合色差
            delta_e = np.sqrt(lightness_diff**2 + chroma_diff**2)
            
            # 6. 分析色差类型
            diff_analysis = {
                'total_score': delta_e,
                'lightness_diff': lightness_diff,
                'chroma_diff': chroma_diff,
                'hue_diff': h_diff,
                'saturation_diff': saturation_diff,
                'dominant_issue': self._identify_dominant_color_issue(
                    lightness_diff, chroma_diff, h_diff, saturation_diff
                )
            }
            
            return diff_analysis
            
        except Exception as e:
            print(f"色差分析失败: {e}")
            return {
                'total_score': 50.0,
                'lightness_diff': 25.0,
                'chroma_diff': 25.0,
                'hue_diff': 15.0,
                'saturation_diff': 20.0,
                'dominant_issue': 'unknown'
            }
    
    def _identify_dominant_color_issue(self, lightness_diff: float, chroma_diff: float, 
                                     hue_diff: float, saturation_diff: float) -> str:
        """识别主要的颜色问题类型"""
        issues = {
            'lightness': lightness_diff,
            'chroma': chroma_diff,
            'hue': hue_diff * 2.0,  # 色调差异权重更高
            'saturation': saturation_diff
        }
        
        dominant_issue = max(issues, key=issues.get)
        return dominant_issue
    
    def _minimal_color_adjustment(self, pred_face: np.ndarray, original_frame: np.ndarray, 
                                bbox: Tuple[int, int, int, int], diff_analysis: dict) -> np.ndarray:
        """最小颜色调整 - 仅微调亮度"""
        try:
            y1, y2, x1, x2 = bbox
            
            # 获取参考区域
            margin = min(15, (y2-y1)//6, (x2-x1)//6)
            ref_y1 = max(0, y1 - margin)
            ref_y2 = min(original_frame.shape[0], y2 + margin)
            ref_x1 = max(0, x1 - margin)
            ref_x2 = min(original_frame.shape[1], x2 + margin)
            
            reference_region = original_frame[ref_y1:ref_y2, ref_x1:ref_x2]
            
            # 仅在LAB空间调整L通道（亮度）
            pred_lab = cv2.cvtColor(pred_face, cv2.COLOR_BGR2LAB).astype(np.float32)
            ref_lab = cv2.cvtColor(reference_region, cv2.COLOR_BGR2LAB)
            
            pred_l_mean = np.mean(pred_lab[:, :, 0])
            ref_l_mean = np.mean(ref_lab[:, :, 0])
            
            # 根据差异程度调整强度
            lightness_diff = diff_analysis.get('lightness_diff', abs(pred_l_mean - ref_l_mean))
            adjustment_strength = min(0.8, lightness_diff / 50.0)
            
            l_adjustment = (ref_l_mean - pred_l_mean) * adjustment_strength
            pred_lab[:, :, 0] = np.clip(pred_lab[:, :, 0] + l_adjustment, 0, 255)
            
            return cv2.cvtColor(pred_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
            
        except Exception as e:
            print(f"最小颜色调整失败: {e}")
            return pred_face
    
    def _targeted_color_adjustment(self, pred_face: np.ndarray, original_frame: np.ndarray, 
                                 bbox: Tuple[int, int, int, int], diff_analysis: dict) -> np.ndarray:
        """针对性颜色调整 - 根据主要问题类型进行调整"""
        try:
            dominant_issue = diff_analysis['dominant_issue']
            
            if dominant_issue == 'lightness':
                return self._adjust_lightness_focused(pred_face, original_frame, bbox, diff_analysis)
            elif dominant_issue == 'hue':
                return self._adjust_hue_focused(pred_face, original_frame, bbox, diff_analysis)
            elif dominant_issue == 'saturation':
                return self._adjust_saturation_focused(pred_face, original_frame, bbox, diff_analysis)
            else:  # chroma or unknown
                return self._adjust_chroma_focused(pred_face, original_frame, bbox, diff_analysis)
                
        except Exception as e:
            print(f"针对性颜色调整失败: {e}")
            return pred_face
    
    def _enhanced_color_adjustment(self, pred_face: np.ndarray, original_frame: np.ndarray, 
                                 bbox: Tuple[int, int, int, int], diff_analysis: dict) -> np.ndarray:
        """强化颜色调整 - 用于严重色差场景，采用渐进式保守调整"""
        try:
            y1, y2, x1, x2 = bbox
            
            # 获取参考区域
            margin = min(25, (y2-y1)//3, (x2-x1)//3)
            ref_y1 = max(0, y1 - margin)
            ref_y2 = min(original_frame.shape[0], y2 + margin)
            ref_x1 = max(0, x1 - margin)
            ref_x2 = min(original_frame.shape[1], x2 + margin)
            
            reference_region = original_frame[ref_y1:ref_y2, ref_x1:ref_x2]
            
            # 渐进式调整策略 - 每步都很保守，避免过度调整
            adjusted_face = pred_face.copy()
            
            # 1. 第一步：非常温和的LAB调整（大幅降低强度）
            adjusted_face = self._progressive_lab_adjustment(adjusted_face, reference_region, diff_analysis, strength=0.3)
            
            # 2. 第二步：有选择性的HSV微调
            if diff_analysis['dominant_issue'] in ['hue', 'saturation']:
                adjusted_face = self._progressive_hsv_adjustment(adjusted_face, reference_region, diff_analysis, strength=0.2)
            
            # 3. 第三步：极轻度的边缘融合
            adjusted_face = self._gentle_edge_smoothing(adjusted_face, pred_face, strength=0.15)
            
            # 4. 第四步：最小化的肤色自然化
            adjusted_face = self._skin_tone_naturalization_gentle(adjusted_face)
            
            return adjusted_face
            
        except Exception as e:
            print(f"强化颜色调整失败: {e}")
            return pred_face
    
    def _adjust_lightness_focused(self, pred_face: np.ndarray, original_frame: np.ndarray, 
                                bbox: Tuple[int, int, int, int], diff_analysis: dict) -> np.ndarray:
        """专注于亮度调整"""
        try:
            y1, y2, x1, x2 = bbox
            margin = min(20, (y2-y1)//4, (x2-x1)//4)
            ref_y1 = max(0, y1 - margin)
            ref_y2 = min(original_frame.shape[0], y2 + margin)
            ref_x1 = max(0, x1 - margin)
            ref_x2 = min(original_frame.shape[1], x2 + margin)
            
            reference_region = original_frame[ref_y1:ref_y2, ref_x1:ref_x2]
            
            # LAB空间亮度调整
            pred_lab = cv2.cvtColor(pred_face, cv2.COLOR_BGR2LAB).astype(np.float32)
            ref_lab = cv2.cvtColor(reference_region, cv2.COLOR_BGR2LAB)
            
            pred_l_mean = np.mean(pred_lab[:, :, 0])
            ref_l_mean = np.mean(ref_lab[:, :, 0])
            
            # 根据差异程度调整强度
            lightness_diff = diff_analysis['lightness_diff']
            adjustment_strength = min(0.8, lightness_diff / 50.0)
            
            l_adjustment = (ref_l_mean - pred_l_mean) * adjustment_strength
            pred_lab[:, :, 0] = np.clip(pred_lab[:, :, 0] + l_adjustment, 0, 100)
            
            return cv2.cvtColor(pred_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
            
        except Exception as e:
            print(f"亮度调整失败: {e}")
            return pred_face
    
    def _adjust_hue_focused(self, pred_face: np.ndarray, original_frame: np.ndarray, 
                          bbox: Tuple[int, int, int, int], diff_analysis: dict) -> np.ndarray:
        """专注于色调调整"""
        try:
            y1, y2, x1, x2 = bbox
            margin = min(20, (y2-y1)//4, (x2-x1)//4)
            ref_y1 = max(0, y1 - margin)
            ref_y2 = min(original_frame.shape[0], y2 + margin)
            ref_x1 = max(0, x1 - margin)
            ref_x2 = min(original_frame.shape[1], x2 + margin)
            
            reference_region = original_frame[ref_y1:ref_y2, ref_x1:ref_x2]
            
            # HSV空间色调调整
            pred_hsv = cv2.cvtColor(pred_face, cv2.COLOR_BGR2HSV).astype(np.float32)
            ref_hsv = cv2.cvtColor(reference_region, cv2.COLOR_BGR2HSV)
            
            pred_h_mean = np.mean(pred_hsv[:, :, 0])
            ref_h_mean = np.mean(ref_hsv[:, :, 0])
            
            # 处理色调环形特性
            h_diff = ref_h_mean - pred_h_mean
            if abs(h_diff) > 90:
                h_diff = h_diff - 180 if h_diff > 0 else h_diff + 180
            
            # 根据差异程度调整强度
            hue_diff = diff_analysis['hue_diff']
            adjustment_strength = min(0.6, hue_diff / 30.0)
            
            h_adjustment = h_diff * adjustment_strength
            pred_hsv[:, :, 0] = (pred_hsv[:, :, 0] + h_adjustment) % 180
            
            return cv2.cvtColor(pred_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            
        except Exception as e:
            print(f"色调调整失败: {e}")
            return pred_face
    
    def _adjust_saturation_focused(self, pred_face: np.ndarray, original_frame: np.ndarray, 
                                 bbox: Tuple[int, int, int, int], diff_analysis: dict) -> np.ndarray:
        """专注于饱和度调整"""
        try:
            y1, y2, x1, x2 = bbox
            margin = min(20, (y2-y1)//4, (x2-x1)//4)
            ref_y1 = max(0, y1 - margin)
            ref_y2 = min(original_frame.shape[0], y2 + margin)
            ref_x1 = max(0, x1 - margin)
            ref_x2 = min(original_frame.shape[1], x2 + margin)
            
            reference_region = original_frame[ref_y1:ref_y2, ref_x1:ref_x2]
            
            # HSV空间饱和度调整
            pred_hsv = cv2.cvtColor(pred_face, cv2.COLOR_BGR2HSV).astype(np.float32)
            ref_hsv = cv2.cvtColor(reference_region, cv2.COLOR_BGR2HSV)
            
            pred_s_mean = np.mean(pred_hsv[:, :, 1])
            ref_s_mean = np.mean(ref_hsv[:, :, 1])
            
            # 根据差异程度调整强度
            saturation_diff = diff_analysis['saturation_diff']
            adjustment_strength = min(0.7, saturation_diff / 40.0)
            
            s_adjustment = (ref_s_mean - pred_s_mean) * adjustment_strength
            pred_hsv[:, :, 1] = np.clip(pred_hsv[:, :, 1] + s_adjustment, 0, 255)
            
            return cv2.cvtColor(pred_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            
        except Exception as e:
            print(f"饱和度调整失败: {e}")
            return pred_face
    
    def _adjust_chroma_focused(self, pred_face: np.ndarray, original_frame: np.ndarray, 
                             bbox: Tuple[int, int, int, int], diff_analysis: dict) -> np.ndarray:
        """专注于色度调整"""
        try:
            y1, y2, x1, x2 = bbox
            margin = min(20, (y2-y1)//4, (x2-x1)//4)
            ref_y1 = max(0, y1 - margin)
            ref_y2 = min(original_frame.shape[0], y2 + margin)
            ref_x1 = max(0, x1 - margin)
            ref_x2 = min(original_frame.shape[1], x2 + margin)
            
            reference_region = original_frame[ref_y1:ref_y2, ref_x1:ref_x2]
            
            # LAB空间色度调整
            pred_lab = cv2.cvtColor(pred_face, cv2.COLOR_BGR2LAB).astype(np.float32)
            ref_lab = cv2.cvtColor(reference_region, cv2.COLOR_BGR2LAB)
            
            pred_a_mean = np.mean(pred_lab[:, :, 1])
            pred_b_mean = np.mean(pred_lab[:, :, 2])
            ref_a_mean = np.mean(ref_lab[:, :, 1])
            ref_b_mean = np.mean(ref_lab[:, :, 2])
            
            # 根据差异程度调整强度
            chroma_diff = diff_analysis['chroma_diff']
            adjustment_strength = min(0.6, chroma_diff / 30.0)
            
            a_adjustment = (ref_a_mean - pred_a_mean) * adjustment_strength
            b_adjustment = (ref_b_mean - pred_b_mean) * adjustment_strength
            
            pred_lab[:, :, 1] = np.clip(pred_lab[:, :, 1] + a_adjustment, -128, 127)
            pred_lab[:, :, 2] = np.clip(pred_lab[:, :, 2] + b_adjustment, -128, 127)
            
            return cv2.cvtColor(pred_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
            
        except Exception as e:
            print(f"色度调整失败: {e}")
            return pred_face
    
    def _enhanced_lab_adjustment(self, pred_face: np.ndarray, reference_region: np.ndarray, 
                               diff_analysis: dict) -> np.ndarray:
        """强化LAB空间调整"""
        try:
            pred_lab = cv2.cvtColor(pred_face, cv2.COLOR_BGR2LAB).astype(np.float32)
            ref_lab = cv2.cvtColor(reference_region, cv2.COLOR_BGR2LAB)
            
            pred_mean = np.mean(pred_lab, axis=(0, 1))
            ref_mean = np.mean(ref_lab, axis=(0, 1))
            
            # 根据分析结果调整各通道强度
            lightness_strength = min(0.8, diff_analysis['lightness_diff'] / 40.0)
            chroma_strength = min(0.7, diff_analysis['chroma_diff'] / 35.0)
            
            # 调整各通道
            pred_lab[:, :, 0] += (ref_mean[0] - pred_mean[0]) * lightness_strength
            pred_lab[:, :, 1] += (ref_mean[1] - pred_mean[1]) * chroma_strength
            pred_lab[:, :, 2] += (ref_mean[2] - pred_mean[2]) * chroma_strength
            
            # 限制范围
            pred_lab[:, :, 0] = np.clip(pred_lab[:, :, 0], 0, 100)
            pred_lab[:, :, 1] = np.clip(pred_lab[:, :, 1], -128, 127)
            pred_lab[:, :, 2] = np.clip(pred_lab[:, :, 2], -128, 127)
            
            return cv2.cvtColor(pred_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
            
        except Exception as e:
            print(f"强化LAB调整失败: {e}")
            return pred_face
    
    def _enhanced_hsv_adjustment(self, pred_face: np.ndarray, reference_region: np.ndarray, 
                               diff_analysis: dict) -> np.ndarray:
        """强化HSV空间调整"""
        try:
            pred_hsv = cv2.cvtColor(pred_face, cv2.COLOR_BGR2HSV).astype(np.float32)
            ref_hsv = cv2.cvtColor(reference_region, cv2.COLOR_BGR2HSV)
            
            pred_mean = np.mean(pred_hsv, axis=(0, 1))
            ref_mean = np.mean(ref_hsv, axis=(0, 1))
            
            # 色调调整（考虑环形特性）
            h_diff = ref_mean[0] - pred_mean[0]
            if abs(h_diff) > 90:
                h_diff = h_diff - 180 if h_diff > 0 else h_diff + 180
            
            hue_strength = min(0.5, diff_analysis['hue_diff'] / 25.0)
            saturation_strength = min(0.6, diff_analysis['saturation_diff'] / 35.0)
            
            # 调整各通道
            pred_hsv[:, :, 0] = (pred_hsv[:, :, 0] + h_diff * hue_strength) % 180
            pred_hsv[:, :, 1] += (ref_mean[1] - pred_mean[1]) * saturation_strength
            
            # 限制范围
            pred_hsv[:, :, 1] = np.clip(pred_hsv[:, :, 1], 0, 255)
            pred_hsv[:, :, 2] = np.clip(pred_hsv[:, :, 2], 0, 255)
            
            return cv2.cvtColor(pred_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            
        except Exception as e:
            print(f"强化HSV调整失败: {e}")
            return pred_face
        """在HSV空间中控制饱和度"""
        try:
            # 转换到HSV
            matched_hsv = cv2.cvtColor(matched_face, cv2.COLOR_BGR2HSV)
            original_hsv = cv2.cvtColor(original_face, cv2.COLOR_BGR2HSV)
            
            # 混合饱和度通道
            original_s = original_hsv[:, :, 1].astype(np.float32)
            matched_s = matched_hsv[:, :, 1].astype(np.float32)
            
            # 计算饱和度比例，防止过度增强
            saturation_ratio = np.where(original_s > 0, matched_s / (original_s + 1e-6), 1.0)
            saturation_ratio = np.clip(saturation_ratio, 0.8, self.max_saturation_boost)
            
            # 混合饱和度
            final_saturation = (original_s * self.saturation_preservation + 
                              matched_s * (1 - self.saturation_preservation))
            
            # 应用饱和度限制
            final_saturation = np.clip(final_saturation, 0, 255)
            matched_hsv[:, :, 1] = final_saturation.astype(np.uint8)
            
            # 转换回BGR
            result = cv2.cvtColor(matched_hsv, cv2.COLOR_HSV2BGR)
            
            return result
            
        except Exception as e:
            print(f"HSV饱和度控制失败: {e}")
            return matched_face
    
    def _analyze_face_colors(self, face_image: np.ndarray) -> dict:
        """分析面部颜色特征"""
        # 转换到多个颜色空间
        lab = cv2.cvtColor(face_image, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
        yuv = cv2.cvtColor(face_image, cv2.COLOR_BGR2YUV)
        
        # 计算颜色统计
        colors = {
            'bgr_mean': np.mean(face_image, axis=(0, 1)),
            'bgr_std': np.std(face_image, axis=(0, 1)),
            'lab_mean': np.mean(lab, axis=(0, 1)),
            'lab_std': np.std(lab, axis=(0, 1)),
            'hsv_mean': np.mean(hsv, axis=(0, 1)),
            'hsv_std': np.std(hsv, axis=(0, 1)),
            'yuv_mean': np.mean(yuv, axis=(0, 1)),
            'yuv_std': np.std(yuv, axis=(0, 1)),
            'brightness': np.mean(lab[:, :, 0]),
            'saturation': np.mean(hsv[:, :, 1]),
        }
        
        return colors

    def _analyze_background_colors(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> dict:
        """分析背景颜色特征"""
        y1, y2, x1, x2 = bbox
        h, w = frame.shape[:2]
        
        # 获取多个参考区域
        regions = []
        
        # 面部周围区域
        margin = min(30, (y2-y1)//3, (x2-x1)//3)
        
        # 上方区域
        if y1 >= margin:
            regions.append(frame[max(0, y1-margin):y1, x1:x2])
        
        # 下方区域
        if y2 + margin < h:
            regions.append(frame[y2:min(h, y2+margin), x1:x2])
        
        # 左侧区域
        if x1 >= margin:
            regions.append(frame[y1:y2, max(0, x1-margin):x1])
        
        # 右侧区域
        if x2 + margin < w:
            regions.append(frame[y1:y2, x2:min(w, x2+margin)])
        
        # 合并所有区域
        if regions:
            combined_region = np.vstack([r.reshape(-1, 3) for r in regions if r.size > 0])
            
            # 转换到多个颜色空间
            lab = cv2.cvtColor(combined_region.reshape(1, -1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
            hsv = cv2.cvtColor(combined_region.reshape(1, -1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
            yuv = cv2.cvtColor(combined_region.reshape(1, -1, 3), cv2.COLOR_BGR2YUV).reshape(-1, 3)
            
            colors = {
                'bgr_mean': np.mean(combined_region, axis=0),
                'bgr_std': np.std(combined_region, axis=0),
                'lab_mean': np.mean(lab, axis=0),
                'lab_std': np.std(lab, axis=0),
                'hsv_mean': np.mean(hsv, axis=0),
                'hsv_std': np.std(hsv, axis=0),
                'yuv_mean': np.mean(yuv, axis=0),
                'yuv_std': np.std(yuv, axis=0),
                'brightness': np.mean(lab[:, 0]),
                'saturation': np.mean(hsv[:, 1]),
            }
        else:
            # 使用整个帧作为参考
            frame_colors = self._analyze_face_colors(frame)
            colors = frame_colors
        
        return colors

    def _intelligent_color_mapping(self, face: np.ndarray, face_colors: dict, bg_colors: dict, strength: float) -> np.ndarray:
        """智能颜色映射"""
        # 在LAB空间进行主要调整
        face_lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # 亮度调整
        brightness_diff = bg_colors['brightness'] - face_colors['brightness']
        brightness_adjustment = brightness_diff * strength * 0.3  # 减少亮度调整强度
        face_lab[:, :, 0] = np.clip(face_lab[:, :, 0] + brightness_adjustment, 0, 255)
        
        # 色彩调整 (A和B通道)
        for i in [1, 2]:  # A, B通道
            color_diff = bg_colors['lab_mean'][i] - face_colors['lab_mean'][i]
            color_adjustment = color_diff * strength * 0.5  # 适度的色彩调整
            face_lab[:, :, i] = np.clip(face_lab[:, :, i] + color_adjustment, 0, 255)
        
        # 转换回BGR
        mapped_bgr = cv2.cvtColor(face_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        
        # 在HSV空间进行饱和度调整
        face_hsv = cv2.cvtColor(mapped_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # 饱和度调整
        saturation_diff = bg_colors['saturation'] - face_colors['saturation']
        saturation_adjustment = saturation_diff * strength * 0.4
        face_hsv[:, :, 1] = np.clip(face_hsv[:, :, 1] + saturation_adjustment, 0, 255)
        
        # 转换回BGR
        result = cv2.cvtColor(face_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return result

    def _edge_color_blending(self, face: np.ndarray, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """边缘颜色融合"""
        y1, y2, x1, x2 = bbox
        
        # 创建边缘权重掩码
        h, w = face.shape[:2]
        edge_width = min(15, h//8, w//8)
        
        # 创建距离变换掩码
        mask = np.ones((h, w), dtype=np.float32)
        mask[:edge_width, :] = 0
        mask[-edge_width:, :] = 0
        mask[:, :edge_width] = 0
        mask[:, -edge_width:] = 0
        
        # 距离变换
        dist_transform = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
        edge_mask = np.clip(dist_transform / edge_width, 0, 1)
        
        # 扩展到3通道
        edge_mask_3d = np.stack([edge_mask] * 3, axis=2)
        
        # 获取对应的原始区域
        original_region = frame[y1:y2, x1:x2]
        
        # 边缘融合
        blended = face * edge_mask_3d + original_region * (1 - edge_mask_3d)
        
        return blended.astype(np.uint8)
    
    def _edge_color_blending_gentle(self, face: np.ndarray, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """温和的边缘颜色融合 - 减少强度"""
        y1, y2, x1, x2 = bbox
        
        # 创建更温和的边缘权重掩码
        h, w = face.shape[:2]
        edge_width = min(10, h//12, w//12)  # 减少边缘宽度
        
        # 创建距离变换掩码
        mask = np.ones((h, w), dtype=np.float32)
        mask[:edge_width, :] = 0
        mask[-edge_width:, :] = 0
        mask[:, :edge_width] = 0
        mask[:, -edge_width:] = 0
        
        # 距离变换
        dist_transform = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
        edge_mask = np.clip(dist_transform / edge_width, 0, 1)
        
        # 应用更平滑的过渡
        edge_mask = np.power(edge_mask, 0.7)  # 使过渡更平滑
        
        # 扩展到3通道
        edge_mask_3d = np.stack([edge_mask] * 3, axis=2)
        
        # 获取对应的原始区域
        original_region = frame[y1:y2, x1:x2]
        
        # 温和的边缘融合
        blended = face * edge_mask_3d + original_region * (1 - edge_mask_3d)
        
        return blended.astype(np.uint8)

    def _skin_tone_naturalization(self, face: np.ndarray) -> np.ndarray:
        """肤色自然化处理"""
        # 转换到YUV空间进行肤色调整
        yuv = cv2.cvtColor(face, cv2.COLOR_BGR2YUV).astype(np.float32)
        
        # 肤色范围调整
        # Y通道 - 亮度保持
        # U通道 - 减少蓝色偏移
        # V通道 - 减少红色偏移
        
        # 轻微调整UV通道，使肤色更自然
        yuv[:, :, 1] = np.clip(yuv[:, :, 1] * 0.95, 0, 255)  # 减少蓝色
        yuv[:, :, 2] = np.clip(yuv[:, :, 2] * 0.98, 0, 255)  # 减少红色
        
        # 转换回BGR
        result = cv2.cvtColor(yuv.astype(np.uint8), cv2.COLOR_YUV2BGR)
        
        return result
    
    def _skin_tone_naturalization_gentle(self, face: np.ndarray) -> np.ndarray:
        """温和的肤色自然化处理 - 减少调整强度"""
        # 转换到YUV空间进行肤色调整
        yuv = cv2.cvtColor(face, cv2.COLOR_BGR2YUV).astype(np.float32)
        
        # 更温和的肤色调整
        # Y通道 - 亮度保持
        # U通道 - 轻微减少蓝色偏移
        # V通道 - 轻微减少红色偏移
        
        # 非常轻微的调整UV通道
        yuv[:, :, 1] = np.clip(yuv[:, :, 1] * 0.98, 0, 255)  # 轻微减少蓝色
        yuv[:, :, 2] = np.clip(yuv[:, :, 2] * 0.99, 0, 255)  # 轻微减少红色
        
        # 转换回BGR
        result = cv2.cvtColor(yuv.astype(np.uint8), cv2.COLOR_YUV2BGR)
        
        return result
    
    def _compensate_lighting_gentle(self, pred_face: np.ndarray, original_frame: np.ndarray, 
                                  bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """温和的光照补偿 - 减少对比度增强"""
        try:
            y1, y2, x1, x2 = bbox
            
            # 计算周围区域的平均亮度
            margin = min(15, (y2-y1)//6, (x2-x1)//6)
            
            surrounding_regions = []
            
            # 上方
            if y1 >= margin:
                surrounding_regions.append(original_frame[y1-margin:y1, x1:x2])
            
            # 下方
            if y2 + margin < original_frame.shape[0]:
                surrounding_regions.append(original_frame[y2:y2+margin, x1:x2])
            
            # 左侧
            if x1 >= margin:
                surrounding_regions.append(original_frame[y1:y2, x1-margin:x1])
            
            # 右侧
            if x2 + margin < original_frame.shape[1]:
                surrounding_regions.append(original_frame[y1:y2, x2:x2+margin])
            
            if not surrounding_regions:
                return pred_face
            
            # 计算周围亮度
            surrounding_brightness = np.mean([np.mean(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)) 
                                            for region in surrounding_regions])
            
            # 计算面部亮度
            face_brightness = np.mean(cv2.cvtColor(pred_face, cv2.COLOR_BGR2GRAY))
            
            # 温和的光照补偿
            if abs(surrounding_brightness - face_brightness) > self.brightness_threshold:
                brightness_ratio = surrounding_brightness / (face_brightness + 1e-6)
                # 更保守的调整范围
                brightness_ratio = np.clip(brightness_ratio, 0.85, self.max_brightness_adjustment)
                
                compensated = pred_face.astype(np.float32) * brightness_ratio
                compensated = np.clip(compensated, 0, 255).astype(np.uint8)
                
                return compensated
            
            return pred_face
            
        except Exception as e:
            print(f"光照补偿失败: {e}")
            return pred_face
class AdvancedFaceBlending:
    """高级面部融合类 - 兼容性包装器"""
    
    def __init__(self):
        self.improved_blender = ImprovedFaceBlending()
        self.fast_blender = FastFaceBlending()
    
    def blend_face(self, *args, **kwargs):
        """
        智能面部融合 - 自动检测参数格式
        
        支持两种调用格式:
        1. lipreal384格式: blend_face(pred_frame, original_frame, bbox)
        2. 传统格式: blend_face(source_face, target_face, landmarks)
        """
        try:
            # 检测调用格式
            if len(args) >= 3:
                # 检查第三个参数是否为bbox (tuple/list of 4 numbers)
                third_arg = args[2]
                if (isinstance(third_arg, (tuple, list)) and 
                    len(third_arg) == 4 and 
                    all(isinstance(x, (int, float)) for x in third_arg)):
                    
                    # lipreal384格式: (pred_frame, original_frame, bbox)
                    pred_frame, original_frame, bbox = args[:3]
                    feather_amount = kwargs.get('feather_amount', 0.8)
                    
                    # 使用改进的融合算法
                    result = self.improved_blender.blend_face(
                        pred_frame, original_frame, bbox, feather_amount
                    )
                    
                    if result is not None:
                        return result
                    else:
                        # 失败时回退到简单替换
                        return self._fallback_blend(pred_frame, original_frame, bbox)
                
                else:
                    # 传统格式: (source_face, target_face, landmarks)
                    source_face, target_face = args[:2]
                    landmarks = args[2] if len(args) > 2 else None
                    feather_amount = kwargs.get('feather_amount', 0.8)
                    
                    # 使用快速融合
                    return self.fast_blender.blend_face_fast(
                        source_face, target_face, landmarks, feather_amount
                    )
            
            # 参数不足，返回None
            return None
            
        except Exception as e:
            print(f"融合过程出错: {e}")
            # 尝试简单回退
            if len(args) >= 2:
                return args[1]  # 返回原始帧
            return None
    
    def _fallback_blend(self, pred_frame, original_frame, bbox):
        """简单回退融合方法"""
        try:
            y1, y2, x1, x2 = bbox
            result = original_frame.copy()
            
            # 调整尺寸并直接替换
            target_width = x2 - x1
            target_height = y2 - y1
            resized_pred = cv2.resize(pred_frame, (target_width, target_height))
            
            result[y1:y2, x1:x2] = resized_pred
            return result
            
        except Exception as e:
            print(f"回退融合失败: {e}")
            return original_frame

# 全局实例
fast_face_blender = FastFaceBlending()
advanced_face_blender = AdvancedFaceBlending()

def get_face_blender(enable_poisson=False, enable_color_matching=False,
                     enable_edge_smoothing=True, enable_noise_reduction=True):
    """获取面部融合器实例
    
    Args:
        enable_poisson: 启用泊松融合 (默认: False)
        enable_color_matching: 启用颜色匹配 (默认: False)
        enable_edge_smoothing: 启用边缘平滑 (默认: True)
        enable_noise_reduction: 启用降噪 (默认: True)
    
    Returns:
        ImprovedFaceBlending: 改进的面部融合器实例
    """
    # 创建配置化的融合器实例
    blender = ImprovedFaceBlending()
    
    # 根据参数配置功能（默认禁用调色/泊松）
    blender.enable_poisson = enable_poisson
    blender.enable_color_matching = enable_color_matching
    blender.color_match_enabled = enable_color_matching  # 映射到内部属性
    blender.enable_edge_smoothing = enable_edge_smoothing
    blender.enable_noise_reduction = enable_noise_reduction
    
    return blender

# 兼容性函数
def blend_face(*args, **kwargs):
    """全局融合函数"""
    return advanced_face_blender.blend_face(*args, **kwargs)

# 历史兼容性
def match_color_histogram(source, target):
    """颜色直方图匹配 - 兼容性函数"""
    # 已禁用调色逻辑，直接返回源图以保持颜色一致性
    return source

def reduce_noise(image, strength=0.5):
    """降噪 - 兼容性函数"""
    if strength > 0:
        return cv2.bilateralFilter(image, 9, 75, 75)
    return image

def enhance_edges(image, strength=0.3):
    """边缘增强 - 兼容性函数"""
    if strength > 0:
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(image, -1, kernel)
        return cv2.addWeighted(image, 1-strength, enhanced, strength, 0)
    return image

if __name__ == "__main__":
    print("改进的面部融合模块已加载")
    print("主要改进:")
    print("✅ 仅遮罩融合 - 保持原视频颜色一致性")
    print("✅ 自适应羽化 - 针对384模型优化")
    print("✅ 向后兼容 - 支持原有调用方式")
