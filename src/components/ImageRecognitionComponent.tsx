"use client"

import React, { useState } from 'react'
import { Box, Title, Text, Paper, Loader, Group, Badge, Card, Image, Progress, Button, FileInput, Stack } from '@mantine/core'
import { detectImage, initializeModel, preprocessImage, type ImageDetectionResult } from '../lib/model/model'

export const ImageRecognitionComponent = () => {
  const [result, setResult] = useState<ImageDetectionResult | null>(null)
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const [modelInitialized, setModelInitialized] = useState<boolean>(false)
  const [uploadedImage, setUploadedImage] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)

  // ファイルアップロード時の処理
  const handleFileUpload = (file: File | null) => {
    setUploadedImage(file);
    setResult(null);
    
    // ファイルが選択された場合、プレビュー用のURLを作成
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    } else {
      setImagePreview(null);
    }
  };

  // 画像認識を実行する関数
  const processUploadedImage = async () => {
    if (!uploadedImage) {
      setError('画像がアップロードされていません');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setResult(null);

      // まだモデルが初期化されていなければ初期化
      if (!modelInitialized) {
        console.log('モデルを初期化中...');
        await initializeModel();
        setModelInitialized(true);
      }

      // ブラウザ環境でも動作するようにDataURLを使用
      const reader = new FileReader();
      const imageDataUrl = await new Promise<string>((resolve, reject) => {
        reader.onload = () => resolve(reader.result as string);
        reader.onerror = reject;
        reader.readAsDataURL(uploadedImage);
      });

      // アップロードされた画像を前処理
      console.log('画像を前処理中...');
      const preprocessedData = await preprocessImage(imageDataUrl);

      // 画像認識を実行
      console.log('画像認識を実行中...');
      const detectionResult = await detectImage(preprocessedData);
      
      // 結果をステートに保存
      setResult(detectionResult);
    } catch (err) {
      console.error('画像処理エラー:', err);
      setError(err instanceof Error ? err.message : '画像処理中にエラーが発生しました');
    } finally {
      setLoading(false);
    }
  }

  const formatScore = (score: number): string => {
    // スコアをパーセント表示に変換（小数点以下2桁）
    return `${(score * 100).toFixed(2)}%`
  }

  return (
    <>
      <Card withBorder shadow="sm" p="lg" radius="md" mb="lg">
        <Stack gap="md">
          <FileInput
            label="画像ファイルを選択"
            placeholder="クリックして画像をアップロード"
            accept="image/png,image/jpeg,image/jpg"
            value={uploadedImage}
            onChange={handleFileUpload}
            clearable
          />
          
          {imagePreview && (
            <Card.Section>
              <Image
                src={imagePreview}
                h={300}
                alt="アップロードされた画像"
                fit="contain"
              />
            </Card.Section>
          )}
          
          <Text size="sm" c="dimmed" mb="md">
            この画像をResNet50モデルで分類します
          </Text>
          
          <Button 
            onClick={processUploadedImage} 
            disabled={loading || !uploadedImage} 
            fullWidth 
            color="blue"
            leftSection={loading ? <Loader size="xs" /> : null}
          >
            {loading ? '処理中...' : '画像解析を開始'}
          </Button>
        </Stack>
      </Card>

      {error && (
        <Paper withBorder p="md" radius="md" bg="red.0">
          <Text c="red">エラー: {error}</Text>
        </Paper>
      )}

      {result && (
        <Paper withBorder p="md" radius="md">
          <Title order={2} size="h3" mb="md">認識結果（上位3件）</Title>
          
          {result.predictions.map((pred, index) => (
            <Box key={index} mb="md">
              <Group justify="space-between" mb={5}>
                <Group>
                  <Badge size="lg">{index + 1}</Badge>
                  <Text fw={500}>{pred.label}</Text>
                </Group>
                <Text>{formatScore(pred.score)}</Text>
              </Group>
              <Progress 
                value={pred.score * 100}
                color={index === 0 ? 'blue' : index === 1 ? 'teal' : 'cyan'} 
                size="lg"
                radius="xl"
                striped
              />
            </Box>
          ))}
          
          <Text size="sm" c="dimmed" mt="lg">
            処理時間: {result.executionTimeMs.toFixed(2)}ms
          </Text>
        </Paper>
      )}
    </>
  )
}