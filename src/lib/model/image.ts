"use client"
import { Jimp, intToRGBA } from 'jimp'

/**
 * Jimp.readが受け付ける入力型（URL、パス、バッファなど）
 * Jimpライブラリの型定義に合わせて柔軟な入力を受け付ける
 */
export type JimpImageInput = Parameters<typeof Jimp.read>[0]

/**
 * ResNet50のための画像前処理を行う関数
 * @param image - Jimpで読み込んだ画像
 * @returns 前処理された画像データ
 */
export const preprocessImage = async (imageInput: JimpImageInput): Promise<Float32Array> => {
  const image = await Jimp.read(imageInput)
  // ResNet50の入力サイズに合わせてリサイズ
  image.resize({ w: 224, h: 224 })

  // ImageNetの平均値と標準偏差
  const mean = [0.485, 0.456, 0.406]
  const std = [0.229, 0.224, 0.225]

  // 関数型アプローチで画像データを処理
  const processPixels = (): Float32Array => {
    // ImageNetの画像サイズ定数
    const width = 224
    const height = 224
    const channels = 3

    // イミュータブルな3次元配列を先に作成
    const pixelData = Array.from({ length: channels }, (_, c) => Array.from({ length: height }, (_, h) => Array.from({ length: width }, (_, w) => {
      const pixelColor = intToRGBA(image.getPixelColor(w, h))
      // RGBチャネルの順番で格納
      const channelValue = c === 0 ? pixelColor.r : (c === 1 ? pixelColor.g : pixelColor.b)
      // 正規化: [0,255] -> [0,1] -> [(x-mean)/std]
      return (channelValue / 255.0 - mean[c]) / std[c]
    })
    )
    )

    // 3次元配列をFloat32Arrayに変換（NCHW形式）
    const result = new Float32Array(channels * height * width)

    // 一度だけ書き込み操作を行う（完全イミュータブルは不可能なので最小限の変更に）
    pixelData.forEach((channelData, c) => {
      channelData.forEach((rowData, h) => {
        rowData.forEach((value, w) => {
          const index = c * height * width + h * width + w
          result[index] = value
        })
      })
    })

    return result
  }

  return processPixels()
}
