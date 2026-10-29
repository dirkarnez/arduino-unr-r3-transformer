#include <Arduino.h>
#include <math.h>

// 超參數定義 (極致微型化)
const int SEQ_LEN = 2; // 序列長度 (2個 Token)
const int D_MODEL = 2; // 每個 Token 的向量維度 (Embedding Size = 2)

// 1. 輸入矩陣 X (2x2) : 代表 [Token_1, Token_2] 的原始詞嵌入向量
float X[SEQ_LEN][D_MODEL] = {
  {1.0, 0.0}, // 第一個單字的語義向量 (例如："小明")
  {0.0, 1.0}  // 第二個單字的語義向量 (例如："小狗")
};

// 2. 權重矩陣 (事先訓練好的 2x2 參數)
float Wq[D_MODEL][D_MODEL] = {{0.5, 0.2}, {0.1, 0.8}};
float Wk[D_MODEL][D_MODEL] = {{0.3, 0.6}, {0.7, 0.2}};
float Wv[D_MODEL][D_MODEL] = {{0.9, 0.1}, {0.4, 0.5}};

// 儲存中間與最終結果的快取
float Q[SEQ_LEN][D_MODEL];
float K[SEQ_LEN][D_MODEL];
float V[SEQ_LEN][D_MODEL];
float Attention_Scores[SEQ_LEN][SEQ_LEN];
float Attention_Weights[SEQ_LEN][SEQ_LEN];
float Output[SEQ_LEN][D_MODEL];

// 矩陣相乘函數：C = A * B
void matrixMultiply(int rowsA, int colsA, int colsB, float A[], float B[], float C[]) {
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
  Serial.println("--- 開始微型 Transformer 演算法計算 (已整合位置編碼) ---");

  // ==========================================
  // 【新步驟】計算並注入正弦/餘弦位置編碼 (Positional Encoding)
  // ==========================================
  Serial.println("\n[1] 正在為輸入矩陣注入位置資訊...");
  for (int pos = 0; pos < SEQ_LEN; pos++) {
    for (int i = 0; i < D_MODEL; i++) {
      float pe_val = 0.0;
      
      // 依據經典公式：偶數維度用 sin，奇數維度用 cos
      if (i % 2 == 0) {
        // 這裡 2i = i，d_model = D_MODEL
        pe_val = sin(pos / pow(10000.0, (float)i / D_MODEL));
      } else {
        // 這裡 2i+1 = i，因此傳入公式的 2i 相當於 i-1
        pe_val = cos(pos / pow(10000.0, (float)(i - 1) / D_MODEL));
      }
      
      // 直接與原始詞向量相加：X_final = X_embed + PE
      X[pos][i] += pe_val;
    }
  }

  // 印出注入位置編碼後的 X 矩陣
  Serial.println("注入位置編碼後的輸入矩陣 X (包含語義+位置):");
  for (int i = 0; i < SEQ_LEN; i++) {
    Serial.print("Token "); Serial.print(i); Serial.print(": [");
    for (int j = 0; j < D_MODEL; j++) {
      Serial.print(X[i][j], 4);
      if (j < D_MODEL - 1) Serial.print(", ");
    }
    Serial.println("]");
  }

  // ==========================================
  // 步驟一：線性投影得到 Query, Key, Value 矩陣
  // ==========================================
  // 此時的 X 已經帶有順序資訊，算出來的 Q, K, V 也將具備位置感知能力
  matrixMultiply(SEQ_LEN, D_MODEL, D_MODEL, X, Wq, Q);
  matrixMultiply(SEQ_LEN, D_MODEL, D_MODEL, X, Wk, K);
  matrixMultiply(SEQ_LEN, D_MODEL, D_MODEL, X, Wv, V);

  // ==========================================
  // 步驟二：計算 Attention Scores (Q 乘以 K^T)
  // ==========================================
  float sqrt_d = sqrt(D_MODEL);
  for (int i = 0; i < SEQ_LEN; i++) {
    for (int j = 0; j < SEQ_LEN; j++) {
      float dot_product = 0;
      for (int k = 0; k < D_MODEL; k++) {
        dot_product += Q[i][k] * K[j][k];
      }
      Attention_Scores[i][j] = dot_product / sqrt_d; // 縮放
    }
  }

  // ==========================================
  // 步驟三：Softmax 歸一化得到注意力權重
  // ==========================================
  for (int i = 0; i < SEQ_LEN; i++) {
    float sum_exp = 0.0;
    for (int j = 0; j < SEQ_LEN; j++) {
      Attention_Weights[i][j] = exp(Attention_Scores[i][j]);
      sum_exp += Attention_Weights[i][j];
    }
    for (int j = 0; j < SEQ_LEN; j++) {
      Attention_Weights[i][j] /= sum_exp;
    }
  }

  // ==========================================
  // 步驟四：權重乘以 Value 矩陣得到最終輸出
  // ==========================================
  for (int i = 0; i < SEQ_LEN; i++) {
    for (int j = 0; j < D_MODEL; j++) {
      Output[i][j] = 0;
      for (int k = 0; k < SEQ_LEN; k++) {
        Output[i][j] += Attention_Weights[i][k] * V[k][j];
      }
    }
  }

  // --- 列印最終結果 ---
  Serial.println("\n[2] Transformer Attention 最終輸出結果 (Output Matrix):");
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
