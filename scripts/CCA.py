# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from pycocotools import mask as mask_utils

# def separate_ignore_regions(rle_mask):
#     """
#     Given an RLE mask representing the ignore region, separate individual ignored objects
#     and return a list of individual masks and their bounding boxes.
#     """
#     # Decode the RLE mask into a binary mask
#     binary_mask = mask_utils.decode(rle_mask)

#     # Find connected components
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=4)

#     # Create separate masks and bounding boxes for each connected component
#     individual_masks = []
#     bounding_boxes = []
#     for label in range(1, num_labels):  # Skip label 0 (background)
#         individual_mask = (labels == label).astype(np.uint8)

#         # Encode individual mask back to RLE
#         rle_individual = mask_utils.encode(np.asfortranarray(individual_mask))
#         individual_masks.append(rle_individual)

#         # Get bounding box for this mask
#         bbox = mask_utils.toBbox(rle_individual)
#         bounding_boxes.append(bbox)

#     return individual_masks, bounding_boxes

# def visualize_and_save_ignore_masks(image, rle_mask, save_path="ignore_masks_visualization.png"):
#     """
#     Visualize separated ignore masks with bounding boxes on the same image and save the output.
#     """
#     # Decode the original ignore mask
#     ignore_mask = mask_utils.decode(rle_mask)

#     # Separate individual ignore regions
#     separated_masks, bboxes = separate_ignore_regions(rle_mask)

#     # Convert image to RGB for Matplotlib display
#     overlay_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

#     # Color map for masks (cycling colors)
#     color_map = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

#     for idx, mask in enumerate(separated_masks):
#         mask_decoded = mask_utils.decode(mask)
#         color = color_map[idx % len(color_map)]  # Cycle through colors

#         # Overlay masks (semi-transparent effect)
#         overlay_image[mask_decoded > 0] = overlay_image[mask_decoded > 0] * 0.5 + np.array(color) * 0.5

#         # Draw bounding boxes
#         x, y, w, h = bboxes[idx]
#         cv2.rectangle(overlay_image, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)

#     # Save the output instead of displaying
#     plt.figure(figsize=(10, 6))
#     plt.imshow(overlay_image)
#     plt.axis("off")
#     plt.title("Ignore Region Masks with BBoxes")
#     plt.savefig(save_path, bbox_inches="tight", dpi=300)  # Save as high-quality image
#     plt.close()  # Close the figure to prevent it from displaying

#     print(f"Visualization saved at {save_path}")

# # Example usage
# # Load an example image (replace with actual image path)
# image = cv2.imread("/cta/users/grad4/master/datasets/MOTS/train/MOTS20-02/img1/000001.jpg")  # Load the image

# a = "no\\<X3`n00000000000000000000000000000000000000000000000000000000000000000000000000000000fg^3YMQ[aL1O1O1O1O1O5K1O1hoNWO_o0i0aPOWO_o0j0`POWO_o0i0`POXO`o0o0YPOQOgo0S1QPOROno0\\101O001O00001O001O001O1O00_OdPOgN[o0V1jPOiNUo0T1oPOkNQo0T1QQOlNnn0S1TQOnNjn0Q1WQOQOgn0m0\\QOROdn0n0S1N2N2M3K5K5000000O1O11O1O2N3M2N4L2N2N2N2N1C75000000d0\\O00000000000000O1O1N2O1000M2100O1000000O10000O1000000TOUPOGoo06bPO\\O^o0b0gPOZOZo0f0l0Ob0^OVdb12i[]N001O0QPOI]n08fQOJTn06PROLjm04m10kPOJhl07\\22N3M4L3M8H1O1lPOQO]m0P1aROHhl0;TSOFll0=QSOCol0=QSOCol0=QSOCol0=QSOCol0=QSOCol0=QSOBPm0?nROBRm0?mRO@G^OVl0S1RTO_OVO0hl0d0oSO]OVO1kl0d0nSO`0Rl0AmSO?Sl0DjSO<Vl0GgSO9Yl0HfSO8Zl0JdSO6gl0AWSO?il0CTSO>ll0DnRO`0Rm0S2000000000000000000000000`0@000000000000000000000000000000000000iS2]NjmM00000SOm00000000000000000000000000000000000000000000000000ZOf00__l0Le`SO000000000000000000000000000000000000000000000000000000000000000000000000000000000000<D000000000000000000000000000008H0000000000000000000000000000000000000000000000000000000000000000000000S\\j0R1kbUO0000000000000000000000000000000000000000000000000000000000[MgQOk1Yn0mMQROQ2mn0N001O001O00001O00O10000000000000000O100O100O1TOgMmQOKNa2Tn0j0000000TOoLSSOo2ml0VMPSOh2Pm0[MmROe2Sm0P100000000000000000000000000000000000000000000000000000000N2000000000bM^200000000PObPOF^o07fPOHZo04kPOKUo02mPOOSo0JTQO6ln0GWQO9in0B\\QO>dn0_OoPO3C=ji?EZWA12N2Nb_90R`Fb0G3G9N2O100O[2hLeLmROM7`3ll0l00000o0QO000000000000000000000000000000000000000000000000000000000005K0000000000000000000002N13MO1O1O1O002N1O1O1O1O00CjLYROU3gm0lL[ROQ3em0oL\\ROP3dm0QM^ROl2bm0TMcROg2]m0YMdSOf1Vo0QO00000000000000000000[`o0l1i]PO000000000000000000000000000000000000000000000000000000000000000fSf3m0]kYL00000000000000000000000000000000000000000000000000000000000000000Vj?\\N^W@0000000000O1O1O1N2O1O1O1N200O100O100O100N200O1O100O1N200O100O100O10000O10000N2O10[RUa0".encode(encoding='UTF-8')
# # Example: Assuming `ignore_rle` is the RLE mask for the ignore region from the MOTS dataset
# ignore_rle = {
#     'size': [1080, 1920],  # Height, Width of the image
#     'counts': a
# }

# visualize_and_save_ignore_masks(image, ignore_rle, save_path="/cta/users/grad4/master/datasets/MOTS/output_ignore_masks_11_111.png")

# import numpy as np
# import cv2
# from pycocotools import mask as mask_utils

# def extract_ignore_objects_to_binary_mask(rle_mask):
#     """
#     Given an RLE mask representing the ignore region, extract the individual objects
#     using connected component analysis and return a single binary mask containing
#     the objects.
#     """
#     # Decode the RLE mask into a binary mask
#     binary_mask = mask_utils.decode(rle_mask)
#     cv2.imwrite('XXX.png', binary_mask.astype(np.uint8))

#     # Find connected components (CCA) in the binary mask
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=4)

#     # Initialize a binary mask to store the ignore region objects (all objects combined)
#     combined_mask = np.zeros_like(binary_mask, dtype=np.uint8)

#     # Iterate over all labels (connected components) to merge them into the final mask
#     for label in range(1, num_labels):  # Skip label 0 (background)
#         # Extract the individual object as a binary mask
#         individual_mask = (labels == label).astype(np.uint8)
        
#         # Add the object to the combined mask (use OR operation to combine masks)
#         combined_mask = np.logical_or(combined_mask, individual_mask).astype(np.uint8)

#     return combined_mask

# def save_ignore_mask_as_binary_image(image, rle_mask, save_path="ignore_objects_binary_mask.png"):
#     """
#     Save the binary mask for the ignored region objects in the image as a single binary mask image.
#     """
#     # Extract the combined binary mask containing all ignored objects
#     combined_mask = extract_ignore_objects_to_binary_mask(rle_mask)

#     # Optionally: you can overlay this mask on the image to visualize it, but here we just save the mask.
#     # Save the combined mask as a binary image
#     cv2.imwrite(save_path, combined_mask * 255)  # Multiply by 255 to save as a white/black image (0 or 255)

#     print(f"Binary mask saved at {save_path}")

# # Example usage
# # Load an example image (replace with actual image path)
# image = cv2.imread("/cta/users/grad4/master/datasets/MOTS/train/MOTS20-02/img1/000001.jpg")  # Load the image

# # Example RLE mask for ignored regions (replace this with the actual RLE mask)
# a = "no\\<X3`n00000000000000000000000000000000000000000000000000000000000000000000000000000000fg^3YMQ[aL1O1O1O1O1O5K1O1hoNWO_o0i0aPOWO_o0j0`POWO_o0i0`POXO`o0o0YPOQOgo0S1QPOROno0\\101O001O00001O001O001O1O00_OdPOgN[o0V1jPOiNUo0T1oPOkNQo0T1QQOlNnn0S1TQOnNjn0Q1WQOQOgn0m0\\QOROdn0n0S1N2N2M3K5K5000000O1O11O1O2N3M2N4L2N2N2N2N1C75000000d0\\O00000000000000O1O1N2O1000M2100O1000000O10000O1000000TOUPOGoo06bPO\\O^o0b0gPOZOZo0f0l0Ob0^OVdb12i[]N001O0QPOI]n08fQOJTn06PROLjm04m10kPOJhl07\\22N3M4L3M8H1O1lPOQO]m0P1aROHhl0;TSOFll0=QSOCol0=QSOCol0=QSOCol0=QSOCol0=QSOCol0=QSOBPm0?nROBRm0?mRO@G^OVl0S1RTO_OVO0hl0d0oSO]OVO1kl0d0nSO`0Rl0AmSO?Sl0DjSO<Vl0GgSO9Yl0HfSO8Zl0JdSO6gl0AWSO?il0CTSO>ll0DnRO`0Rm0S2000000000000000000000000`0@000000000000000000000000000000000000iS2]NjmM00000SOm00000000000000000000000000000000000000000000000000ZOf00__l0Le`SO000000000000000000000000000000000000000000000000000000000000000000000000000000000000<D000000000000000000000000000008H0000000000000000000000000000000000000000000000000000000000000000000000S\\j0R1kbUO0000000000000000000000000000000000000000000000000000000000[MgQOk1Yn0mMQROQ2mn0N001O001O00001O00O10000000000000000O100O100O1TOgMmQOKNa2Tn0j0000000TOoLSSOo2ml0VMPSOh2Pm0[MmROe2Sm0P100000000000000000000000000000000000000000000000000000000N2000000000bM^200000000PObPOF^o07fPOHZo04kPOKUo02mPOOSo0JTQO6ln0GWQO9in0B\\QO>dn0_OoPO3C=ji?EZWA12N2Nb_90R`Fb0G3G9N2O100O[2hLeLmROM7`3ll0l00000o0QO000000000000000000000000000000000000000000000000000000000005K0000000000000000000002N13MO1O1O1O002N1O1O1O1O00CjLYROU3gm0lL[ROQ3em0oL\\ROP3dm0QM^ROl2bm0TMcROg2]m0YMdSOf1Vo0QO00000000000000000000[`o0l1i]PO000000000000000000000000000000000000000000000000000000000000000fSf3m0]kYL00000000000000000000000000000000000000000000000000000000000000000Vj?\\N^W@0000000000O1O1O1N2O1O1O1N200O100O100O100N200O1O100O1N200O100O100O10000O10000N2O10[RUa0".encode(encoding='UTF-8')

# # Example: Assuming `ignore_rle` is the RLE mask for the ignore region from the MOTS dataset
# ignore_rle = {
#     'size': [1080, 1920],  # Height, Width of the image
#     'counts': a
# }

# # Save the binary mask as a single image containing all ignored objects
# save_ignore_mask_as_binary_image(image, ignore_rle, save_path="output_ignore_objects_binary_mask.png")

### NOTE: Bu kısım ile RLE'yi decode ediyoruz!
# import numpy as np
# import cv2
# from pycocotools import mask as mask_utils

# def decode_and_save_rle_mask(rle_mask, image, mask_output_path="decoded_mask.png", overlay_output_path="image_with_overlay.png"):
#     binary_mask = mask_utils.decode(rle_mask)
#     binary_mask = (binary_mask * 255).astype(np.uint8)
#     mask_rgb = cv2.merge([binary_mask, binary_mask, binary_mask])
#     combined_image = cv2.addWeighted(image, 0.7, mask_rgb, 0.3, 0)
    
#     cv2.imwrite(mask_output_path, binary_mask)
#     cv2.imwrite(overlay_output_path, combined_image)
    
#     print(f"Decoded mask saved at: {mask_output_path}")
#     print(f"Image with overlay saved at: {overlay_output_path}")

# a = "no\\<X3`n00000000000000000000000000000000000000000000000000000000000000000000000000000000fg^3YMQ[aL1O1O1O1O1O5K1O1hoNWO_o0i0aPOWO_o0j0`POWO_o0i0`POXO`o0o0YPOQOgo0S1QPOROno0\\101O001O00001O001O001O1O00_OdPOgN[o0V1jPOiNUo0T1oPOkNQo0T1QQOlNnn0S1TQOnNjn0Q1WQOQOgn0m0\\QOROdn0n0S1N2N2M3K5K5000000O1O11O1O2N3M2N4L2N2N2N2N1C75000000d0\\O00000000000000O1O1N2O1000M2100O1000000O10000O1000000TOUPOGoo06bPO\\O^o0b0gPOZOZo0f0l0Ob0^OVdb12i[]N001O0QPOI]n08fQOJTn06PROLjm04m10kPOJhl07\\22N3M4L3M8H1O1lPOQO]m0P1aROHhl0;TSOFll0=QSOCol0=QSOCol0=QSOCol0=QSOCol0=QSOCol0=QSOBPm0?nROBRm0?mRO@G^OVl0S1RTO_OVO0hl0d0oSO]OVO1kl0d0nSO`0Rl0AmSO?Sl0DjSO<Vl0GgSO9Yl0HfSO8Zl0JdSO6gl0AWSO?il0CTSO>ll0DnRO`0Rm0S2000000000000000000000000`0@000000000000000000000000000000000000iS2]NjmM00000SOm00000000000000000000000000000000000000000000000000ZOf00__l0Le`SO000000000000000000000000000000000000000000000000000000000000000000000000000000000000<D000000000000000000000000000008H0000000000000000000000000000000000000000000000000000000000000000000000S\\j0R1kbUO0000000000000000000000000000000000000000000000000000000000[MgQOk1Yn0mMQROQ2mn0N001O001O00001O00O10000000000000000O100O100O1TOgMmQOKNa2Tn0j0000000TOoLSSOo2ml0VMPSOh2Pm0[MmROe2Sm0P100000000000000000000000000000000000000000000000000000000N2000000000bM^200000000PObPOF^o07fPOHZo04kPOKUo02mPOOSo0JTQO6ln0GWQO9in0B\\QO>dn0_OoPO3C=ji?EZWA12N2Nb_90R`Fb0G3G9N2O100O[2hLeLmROM7`3ll0l00000o0QO000000000000000000000000000000000000000000000000000000000005K0000000000000000000002N13MO1O1O1O002N1O1O1O1O00CjLYROU3gm0lL[ROQ3em0oL\\ROP3dm0QM^ROl2bm0TMcROg2]m0YMdSOf1Vo0QO00000000000000000000[`o0l1i]PO000000000000000000000000000000000000000000000000000000000000000fSf3m0]kYL00000000000000000000000000000000000000000000000000000000000000000Vj?\\N^W@0000000000O1O1O1N2O1O1O1N200O100O100O100N200O1O100O1N200O100O100O10000O10000N2O10[RUa0".encode(encoding='UTF-8')
# ignore_rle = {
#     'size': [1080, 1920],
#     'counts': a
# }

# image = cv2.imread("/cta/users/grad4/master/datasets/MOTS/train/MOTS20-02/img1/000001.jpg")
# decode_and_save_rle_mask(ignore_rle, image, mask_output_path="decoded_mask.png", overlay_output_path="image_with_overlay.png")

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pycocotools import mask as mask_utils

def separate_ignore_regions(rle_mask, min_bbox_area=161):
    binary_mask = mask_utils.decode(rle_mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=4)

    individual_masks, bounding_boxes = [], []
    for label in range(1, num_labels):
        individual_mask = (labels == label).astype(np.uint8)
        rle_individual = mask_utils.encode(np.asfortranarray(individual_mask))
        bbox = mask_utils.toBbox(rle_individual)
        
        x, y, w, h = bbox
        area = w * h
        print(area)
        if area >= min_bbox_area:
            individual_masks.append(rle_individual)
            bounding_boxes.append(bbox)

        # individual_masks.append(rle_individual)
        # bounding_boxes.append(bbox)

    return individual_masks, bounding_boxes

def visualize_and_save_ignore_masks(image, rle_mask, save_path="ignore_masks_visualization.png"):
    ignore_mask = mask_utils.decode(rle_mask)
    separated_masks, bboxes = separate_ignore_regions(rle_mask)
    overlay_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    color_map = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

    for idx, mask in enumerate(separated_masks):
        mask_decoded = mask_utils.decode(mask)
        color = color_map[idx % len(color_map)]

        overlay_image[mask_decoded > 0] = overlay_image[mask_decoded > 0] * 0.5 + np.array(color) * 0.5

        x, y, w, h = bboxes[idx]
        cv2.rectangle(overlay_image, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)

    plt.figure(figsize=(10, 6))
    plt.imshow(overlay_image)
    plt.axis("off")
    plt.title("Ignore Region Masks with BBoxes")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Visualization saved at {save_path}")

image = cv2.imread("/cta/users/grad4/master/datasets/MOTS/train/MOTS20-05/img1/000001.jpg")
rle__ = "ZX77c>8H8H8J6J6eChNg:\\1REjNk:Z1oDlNl:X1oDnNn:U1kDQOR;a2L2N3N2N2O100O10oLiDY2o0SMT9c0nEa2g0oLY9>PFo2;eLd9<PFX33^Lm99PFh3o9WLRFk3m9TLRFn3m9RLSFo3m9PLSFR4k9mKUFU4k9jKUFW4j9iKVFX4Y9aKPG6GZ4U9jKlFNOW4V9[LjFe3V9[LjFb2f0jL_8e0kFa2Y:^MgEc2X:]MhEd2W:]MiEc2W:\\MiEe2V:[MjEf2V:ZMjEg2U:XMkEi2U:WMkEi2U:VMkEP3P:PMoEY33ZLQ9=lFn3T9QLlFP4U9oKkFQ4U9mKlFT4T9lKmFT4R9lKnFS4S9mKmFS4S9nKlFQ4U9oKlFP4U9oKlFY30_LW96iFX35`LT96gFR3c0bLg89hFU3d0^Lg89hFX3e0[L[9d3Q101N3N2N2M3eN]DUOf;h0`DROb;l0cDnN`;Q1eDiN];T1iDfNZ;X1kDbN`;T1fDfNe;o0Z1L5I7IeSQ8".encode(encoding='UTF-8')
ignore_rle = {
    'size': [480, 640], 
    'counts': rle__
}

visualize_and_save_ignore_masks(image, ignore_rle, save_path="output_masks_000001.png")
