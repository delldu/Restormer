import image_clean

image_clean.defocus_predict("images/defocus/*.png", "output/defocus")
image_clean.denoise_predict("images/denoise/*.png", "output/denoise")
image_clean.deblur_predict("images/deblur/*.png", "output/deblur")

# image_clean.derain_predict("images/derain/*.png", "output/derain")
