#functions for white chocolates

#main function is segment_white_chocos -- which runs the following
#mask the background to remove unwanted --> mask and get the white --> detect contours

def mask_background(img_rgb, threshs=[100,100,100], hue_thresh_max=0.3):
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mask_rgb = np.ones_like(img_rgb[:,:,0])
    for i in range(3):
        mask_rgb = np.logical_and(mask_rgb,img_rgb[:,:,i]>threshs[i])
    mask_rgb = np.logical_and(mask_rgb,img_hsv[:,:,0]/255 < hue_thresh_max)
    masked_bg = np.zeros_like(img_rgb)
    for i in range(3):
        masked_bg[:,:,i] = mask_rgb*img_rgb[:,:,i]
    return masked_bg, mask_rgb

def mask_hsv_white(img_rgb, val_sq_thresh_min=0.65,sat_thresh_max=0.1):
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    #value thresholding
    val_choco=img_hsv[:,:,2]/255
    val_choco_thresh = val_choco*val_choco > val_sq_thresh_min
    #saturation thresholding
    sat_choco = img_hsv[:,:,1]/255
    sat_choco_thresh = sat_choco< sat_thresh_max

    #combine to get white chocolate mask
    arr = np.clip(val_choco_thresh+1-sat_choco_thresh,0,1)
    masked_choco = np.zeros_like(img_rgb)
    for i in range(3):
        masked_choco[:,:,i] = arr*img_rgb[:,:,i]
    return masked_choco, arr

def get_contours(masked_choco, area_min=20000, area_max = 200000):
    image = masked_choco.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (71, 71), 0)
    _, thresh = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=2)

    # Step 4: Get sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Step 5: Get sure foreground area (using distance transform)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.6 * dist_transform.max(), 255, 0)

    # Step 6: Find unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Step 7: Label markers
    _, markers = cv2.connectedComponents(sure_fg)

    # Add 1 to all labels so sure background is not 0
    markers = markers + 1
    markers[unknown == 255] = 0

    # Step 8: Apply Watershed
    markers = cv2.watershed(image, markers)

    # Step 9: Extract contours from watershed regions
    output = image.copy()
    output[markers == -1] = [0, 0, 255]  # boundary color

    # Optional: create mask and find contours
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[markers > 1] = 255

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    n=0
    # Filter based on circularity
    contours_detect = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        if area < area_min:
            continue
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        convexity = area / hull_area
    
        if (circularity > 0.3 or convexity>0.6): #and area < area_max:
            n=n+1
            contours_detect.append(cnt)
    return contours_detect, n

def segment_white_chocos(img_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    masked_bg,_ = mask_background(img_rgb)
    masked_choco,_ = mask_hsv_white(masked_bg)
    contours_detected, n  = get_contours(masked_choco)
    cv2.drawContours(img_rgb, contours_detected, -1, (255, 0, 0), 5)
    plt.imshow(img_rgb)
    plt.title(f"Detected {n} contours in {img_path.split('L')[-1]}")
    plt.show()
    return contours_detected, masked_choco
    