import cv2
import numpy as np


def _opt_flow_warp(img1, img2, from_up=False):
    '''
    returns a generated img2 alternative, whose down(up) portion is img2,
    up(down) portion is warped to img1.
    '''
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        img1_gray, img2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    flow[np.where(np.abs(flow) > 20)] = 0

    height, width = flow.shape[:2]
    R2 = np.dstack(np.meshgrid(np.arange(width), np.arange(height)))
    if from_up:
        flow_smooth = flow * np.linspace(0, 1, height).reshape(-1, 1, 1)
    else:
        flow_smooth = flow * np.linspace(1, 0, height).reshape(-1, 1, 1)
    pixel_map = (R2 + flow_smooth).astype(np.float32)
    gen_img2 = cv2.remap(img2, pixel_map[..., 0], pixel_map[..., 1],
                         cv2.INTER_CUBIC,
                         cv2.BORDER_REFLECT)
    return gen_img2


def flow_warp(top_frame, bot_frame, n=3):
    for i in range(n):
        bot_frame_ = _opt_flow_warp(top_frame, bot_frame, False)
        top_frame_ = _opt_flow_warp(bot_frame_, top_frame, True)
        top_frame, bot_frame = top_frame_, bot_frame_
    return top_frame, bot_frame


if __name__ == "__main__":
    img1 = cv2.imread('overlap1.jpg')
    img2 = cv2.imread('overlap2.jpg')

    frame1 = img1.copy()
    frame2 = img2.copy()

    cv2.imwrite('gen_img1.jpg', frame1)
    cv2.imwrite('gen_img2.jpg', frame2)
    cv2.imwrite('diff_12.jpg', cv2.absdiff(frame1, img2))
    cv2.imwrite('diff_21.jpg', cv2.absdiff(frame2, img1))
    cv2.imwrite('diff.jpg', cv2.absdiff(frame1, frame2))
