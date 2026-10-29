#include <Arduino.h>
#include <math.h>

// 超參數定義 (極致微型化)
const int SEQ_LEN = 2; // 序列長度 (2個 Token)
const int D_MODEL = 2; // 每個 Token 的向量維度 (Embedding Size = 2)

// 1. 輸入矩陣 X (2x2) : 代表 [Token_1, Token_2]
float X[SEQ_LEN][D_MODEL] = {
  {1.0, 0.0}, // 第一個單字的向量
  {0.0, 1.0}  // 第二個單字的向量
};

// 2. 權重矩陣 (這裡簡化讓 Wq, Wk, Wv 均為 2x2 矩陣)
// 實務上這些是訓練好的參數，這裡給予固定初始值
float Wq[D_MODEL][D_MODEL] = {{0.5, 0.2}, {0.1, 0.8}};
float Wk[D_MODEL][D_MODEL] = {{0.3, 0.6}, {0.7, 0.2}};
float Wv[D_MODEL][D_MODEL] = {{0.9, 0.1}, {0.4, 0.5}};

// 儲存中間與最終結果的快取 (總計僅佔用幾十個位元組)
float Q[SEQ_LEN][D_MODEL];
float K[SEQ_LEN][D_MODEL];
float V[SEQ_LEN][D_MODEL];
float Attention_Scores[SEQ_LEN][SEQ_LEN];
float Attention_Weights[SEQ_LEN][SEQ_LEN];
float Output[SEQ_LEN][D_MODEL];

// 矩陣相乘函數：C = A * B
void matrixMultiply(int rowsA, int colsA, int colsB, float A[][2], float B[][2], float C[][2]) {
  for (int i = 0; i < rowsA; i++) {
    for (int j = 0; j < colsB; j++) {
      C[i][j] = 0;
      for (int k = 0; k < colsA; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

void setup() {
  Serial.begin(9600);
  while (!Serial);
  Serial.println("--- 開始微型 Transformer 演算法計算 ---");

  // 步驟一：線性投影得到 Query, Key, Value 矩陣
  matrixMultiply(SEQ_LEN, D_MODEL, D_MODEL, X, Wq, Q);
  matrixMultiply(SEQ_LEN, D_MODEL, D_MODEL, X, Wk, K);
  matrixMultiply(SEQ_LEN, D_MODEL, D_MODEL, X, Wv, V);

  // 步驟二：計算 Attention Scores (Q 乘以 K 的轉置矩陣 K^T)
  // Scores[i][j] = Q[i] 點積 K[j]
  float sqrt_d = sqrt(D_MODEL);
  for (int i = 0; i < SEQ_LEN; i++) {
    for (int j = 0; j < SEQ_LEN; j++) {
      float dot_product = 0;
      for (int k = 0; k < D_MODEL; k++) {
        dot_product += Q[i][k] * K[j][k];
      }
      // 縮放 (Scaled Dot-Product)
      Attention_Scores[i][j] = dot_product / sqrt_d;
    }
  }

  // 步驟三：對每一行進行 Softmax 歸一化，得到注意力權重 (Weights)
  for (int i = 0; i < SEQ_LEN; i++) {
    float sum_exp = 0.0;
    // 計算 exp
    for (int j = 0; j < SEQ_LEN; j++) {
      Attention_Weights[i][j] = exp(Attention_Scores[i][j]);
      sum_exp += Attention_Weights[i][j];
    }
    // 歸一化
    for (int j = 0; j < SEQ_LEN; j++) {
      Attention_Weights[i][j] /= sum_exp;
    }
  }

  // 步驟四：權重乘以 Value 矩陣，得到最終輸出 (Output)
  for (int i = 0; i < SEQ_LEN; i++) {
    for (int j = 0; j < D_MODEL; j++) {
      Output[i][j] = 0;
      for (int k = 0; k < SEQ_LEN; k++) {
        Output[i][j] += Attention_Weights[i][k] * V[k][j];
      }
    }
  }

  // --- 列印最終經過「自注意力」加權後的 Output 矩陣 ---
  Serial.println("Transformer Attention 輸出結果 (Output Matrix):");
  for (int i = 0; i < SEQ_LEN; i++) {
    Serial.print("Token "); Serial.print(i); Serial.print(": [");
    for (int j = 0; j < D_MODEL; j++) {
      Serial.print(Output[i][j], 4);
      if (j < D_MODEL - 1) Serial.print(", ");
    }
    Serial.println("]");
  }
}

void loop() {
  // 僅需執行一次
}
