import image_clean

# image_clean.defocus_client("TAI", "images/defocus/*.png", "output/defocus")
# image_clean.defocus_server("TAI")
image_clean.defocus_predict("images/defocus/*.png", "output/defocus")

# image_clean.denoise_client("TAI", "images/denoise/*.png", "output/denoise")
# image_clean.denoise_server("TAI")
image_clean.denoise_predict("images/denoise/*.png", "output/denoise")

# image_clean.deblur_client("TAI", "images/deblur/*.png", "output/deblur")
# image_clean.deblur_server("TAI")
image_clean.deblur_predict("images/deblur/*.png", "output/deblur")

# image_clean.derain_client("TAI", "images/derain/*.png", "output/derain")
# image_clean.derain_server("TAI")
image_clean.derain_predict("images/derain/1*.png", "output/derain")
