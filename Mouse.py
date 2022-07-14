# coding: utf-8
def mouse(img):
    import cv2



    # print img.shape

    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("(%d,%d)" % (x, y))
            xy = "%d,%d" % (x, y)
            cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
            cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 0), thickness=1)
            cv2.imshow("image", img)


    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image", img)

    while (True):
        try:
            cv2.waitKey(100)
        except Exception:
            cv2.destroyAllWindows()
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()
