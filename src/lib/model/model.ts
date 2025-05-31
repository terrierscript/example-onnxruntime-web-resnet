"use client"

import * as ort from 'onnxruntime-web/webgpu'
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
    })))

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

// モデルデータをキャッシュ
let cachedModelData: Uint8Array | null = null

const getModelData = async (): Promise<Uint8Array> => {
  // キャッシュがあればそれを返す
  if (cachedModelData) {
    console.log('キャッシュからモデルデータを取得')
    return cachedModelData
  }
  console.log('あらたにmodelを作成します')

  try {
    // モデルファイルをfetchで取得してUint8Arrayに変換
    const modelPath = '/model/resnet50-v1-12.onnx'
    const modelResponse = await fetch(modelPath)
    const modelArrayBuffer = await modelResponse.arrayBuffer()
    const modelData = new Uint8Array(modelArrayBuffer)

    // キャッシュに保存
    cachedModelData = modelData

    return modelData
  } catch (error) {
    console.error('モデルデータ取得エラー:', error)
    throw error
  }
}

/**
 * モデルデータを取得する関数
 * データは一度だけ取得してキャッシュする
 * @returns モデルデータ
 */
export const initializeModel = async () => {
  await getModelData()

  // 今後の高速化のために空セッション作成
  await createModelSession()

}

/**
 * 新しいONNXセッションを作成する関数
 * @returns 初期化されたモデルセッション
 */
const createModelSession = async (): Promise<ort.InferenceSession> => {
  try {
    // モデルデータを取得
    const modelData = await getModelData()

    // セッションオプションを設定 - WebGPUを使用
    const sessionOptions: ort.InferenceSession.SessionOptions = {
      executionProviders: ['webgpu'],
      graphOptimizationLevel: 'all',
      executionMode: 'parallel', // 並列実行モード
      intraOpNumThreads: 4, // オペレーション内の並列処理スレッド数
      interOpNumThreads: 2, // オペレーション間の並列スレッド数
      enableCpuMemArena: true, // メモリアリーナを有効化
      enableMemPattern: true, // メモリパターンを有効化
    }

    const session = await ort.InferenceSession.create(modelData, sessionOptions)

    return session
  } catch (error) {
    console.error('モデルセッションの初期化エラー:', error)
    throw error
  }
}

/**
 * ResNet50モデルを使って画像データを処理し、出力データを取得する共通関数
 * @param preprocessedData - 前処理済みの画像データ
 * @returns モデルの出力データと実行時間
 */
const runResNetModel = async (preprocessedData: Float32Array): Promise<{ outputData: number[], executionTimeMs: number }> => {
  // 各実行ごとに新しいセッションを作成
  const session = await createModelSession()

  try {
    // パフォーマンス計測開始
    const startTime = performance.now()

    // 推論実行
    const inputTensor = new ort.Tensor('float32', preprocessedData, [1, 3, 224, 224])
    const feeds = { data: inputTensor }
    const results = await session.run(feeds)

    // パフォーマンス計測終了と出力
    const endTime = performance.now()
    const executionTimeMs = endTime - startTime
    console.log(`推論実行時間: ${executionTimeMs}ms`)

    // 出力テンソルから結果を取得
    const outputTensor = results.resnetv17_dense0_fwd
    if (!outputTensor || !('data' in outputTensor)) {
      throw new Error('出力テンソルの形式が不正です')
    }

    // データの型を確認して安全に扱う
    if (!(outputTensor.data instanceof Float32Array)) {
      throw new Error('出力データが予期しない型です')
    }

    // 出力データと実行時間を返却
    return {
      outputData: Array.from(outputTensor.data),
      executionTimeMs
    }
  } catch (error) {
    console.error('ResNetモデル実行エラー:', error)
    throw error
  }
}


/**
 * 入力配列に対してsoftmax計算を行い確率分布を返す関数
 * @param logits - モデル出力の生のスコア配列
 * @returns 確率分布（合計が1になる）
 */
const computeSoftmax = (logits: number[]): number[] => {
  // オーバーフロー対策：最大値を引く
  const maxLogit = Math.max(...logits)
  const expScores = logits.map(logit => Math.exp(logit - maxLogit))
  const sumExpScores = expScores.reduce((sum, expScore) => sum + expScore, 0)

  // 確率分布に変換
  return expScores.map(expScore => expScore / sumExpScores)
}

/**
 * ResNet50モデルの予測結果を表す型
 * 画像分類の結果と確信度を含む
 */
export type ImagePrediction = {
  label: string    // 予測されたラベル
  score: number    // 確信度スコア（0〜1）
  index: number    // ラベルのインデックス
}

/**
 * 画像認識結果を表す型
 */
export type ImageDetectionResult = {
  predictions: ImagePrediction[]  // 上位の予測結果
  executionTimeMs: number         // 実行時間（ms）
}

// ラベルデータをインポート
import labels from '../constants/labels.json'

export const detectImage = async (preprocessedData: Float32Array): Promise<ImageDetectionResult | null> => {
  try {
    // パフォーマンス計測開始
    const startTime = performance.now()

    // 共通関数を使用して出力データを取得
    const { outputData, executionTimeMs } = await runResNetModel(preprocessedData)

    // Softmax計算によって確率分布を取得
    const probabilities = computeSoftmax(outputData)

    // スコアとインデックスのペアを作成
    const indexedScores = probabilities.map((score, index) => ({ score, index }))

    // スコアの降順でソート
    const sortedResults = indexedScores.sort((a, b) => b.score - a.score)

    // 上位3つの結果を取得
    const topPredictions: ImagePrediction[] = sortedResults.slice(0, 10).map(result => ({
      label: labels[result.index],
      score: result.score,
      index: result.index
    }))

    const endTime = performance.now()
    const totalProcessingTimeMs = endTime - startTime

    // 結果をログ出力
    console.log('画像認識結果:', topPredictions)
    console.log(`全処理時間: ${totalProcessingTimeMs}ms（推論実行: ${executionTimeMs}ms）`)

    return {
      predictions: topPredictions,
      executionTimeMs: totalProcessingTimeMs
    }
  } catch (error) {
    console.error('画像認識エラー:', error)
    return null
  }
}