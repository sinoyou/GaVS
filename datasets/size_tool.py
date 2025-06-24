import torch
import torch.nn.functional as F

def rescale_and_crop_field(img, target_size, K_ndc, mask=None):
    ret_list = []

    _, h_in, w_in = img.shape
    h_out, w_out = target_size
    assert h_out <= h_in and w_out <= w_in, f"Target size {target_size} larger than input size {img.shape}"

    # rescale
    scale_factor = max(h_out / h_in, w_out / w_in)
    h_scaled, w_scaled = round(h_in * scale_factor), round(w_in * scale_factor)
    assert h_scaled == h_out or w_scaled == w_out, f"Rescaled size {h_scaled, w_scaled} not statisfy either width or height in the target size {target_size}"
    # img_pil = T.ToPILImage()(img)
    # rescale by opencv resize with inter-area mode
    # img_rescaled = T.ToTensor()(cv2.resize(np.array(img_pil), (w_scaled, h_scaled), interpolation=cv2.INTER_AREA))
    img_rescaled = F.interpolate(img.unsqueeze(0), (h_scaled, w_scaled), mode='bilinear', align_corners=False, antialias=True).squeeze(0)

    # crop
    row, coloum = (h_scaled - h_out) // 2, (w_scaled - w_out) // 2
    img_cropped = img_rescaled[:, row:row+h_out, coloum:coloum+w_out]
    K_ndc_crop = K_ndc.clone()
    K_ndc_crop[0, 0] *= w_scaled / w_out
    K_ndc_crop[1, 1] *= h_scaled / h_out

    ret_list.append(img_cropped)
    ret_list.append(K_ndc_crop)

    if mask is not None:
        mask = F.interpolate(mask.unsqueeze(0), (h_scaled, w_scaled), mode='bilinear', align_corners=False).squeeze(0)
        mask = mask[:, row:row+h_out, coloum:coloum+w_out]
        ret_list.append(mask)
    
    return ret_list