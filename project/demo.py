import image_clean

image_clean.defocus_predict("images/defocus/*.png", "output/defocus")
image_clean.denoise_predict("images/denoise/*.png", "output/denoise")
image_clean.deblur_predict("images/deblur/*.png", "output/deblur")
image_clean.derain_predict("images/derain/*.png", "output/derain")

# demo
# image_clean.defocus_predict("images/Single_Image_Defocus_Deblurring/*.png", "output/demo/defocus")
# # image_clean.defocus2_predict("images/Dual_Pixel_Defocus_Deblurring/1P0A2514*.png", "output/demo/defocus2")
# image_clean.denoise_predict("images/noisy15/*.png", "output/demo/denoise")
# image_clean.denoise_predict_add_noise("images/McMaster/*.tif", "output/demo/denoise")
# image_clean.deblur_predict("images/GoPro_Deblur/G*_11_[0-1]*-000*0[0-1].png", "output/demo/deblur")

