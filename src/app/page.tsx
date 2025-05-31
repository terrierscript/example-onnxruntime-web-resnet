"use client"
import React from 'react'
import { Box, Container, Title } from '@mantine/core'
import dynamic from 'next/dynamic'

// 画像認識コンポーネントをクライアントサイドのみで動的に読み込む
const DynamicImageRecognition = dynamic(
  () => import('../components/ImageRecognitionComponent').then(mod => ({ default: mod.ImageRecognitionComponent })),
  { 
    loading: () => <p>モデルを読み込み中...</p>,
    ssr: false // サーバーサイドレンダリングを無効化
  }
)

export default function Home() {
  return (
    <Box py="md">
      <Container>
        <Title order={1} mb="lg">画像認識デモ</Title>
        <DynamicImageRecognition />
      </Container>
    </Box>
  )
}
