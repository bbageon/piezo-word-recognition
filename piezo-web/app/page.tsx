"use client";

import { useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type SensorData = {
  time: number[];
  value1: number[];
  value2: number[];
};

type PredictResponse = {
  prediction?: string;
  prompt_used?: string;
};

const SAMPLE_RATE_HZ = 200;
const RECORD_DURATION_MS = 1500;
const TOTAL_SAMPLES = (SAMPLE_RATE_HZ * RECORD_DURATION_MS) / 1000;

// ✅ ESP32가 Wi-Fi로 띄운 HTTP 서버 주소(포트 80)
const ESP32_BASE_URL = "http://172.20.10.2:80";

// FastAPI (local)
const FASTAPI_URL = "http://127.0.0.1:8000/predict_llm";
// const FASTAPI_URL = "http://172.20.10.13:8000/predict_llm";


async function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

export default function Home() {
  const [rawData, setRawData] = useState<SensorData | null>(null);
  const [prediction, setPrediction] = useState("");
  const [connected, setConnected] = useState(false);

  const [status, setStatus] = useState("Not connected");
  const [measuring, setMeasuring] = useState(false);
  const [predicting, setPredicting] = useState(false);

  const chartData = useMemo(() => {
    if (!rawData) return [];
    return rawData.time.map((t, i) => ({
      time: t,
      left: rawData.value1[i],
      right: rawData.value2[i],
    }));
  }, [rawData]);

  // ✅ Wi-Fi 연결(실제로는 "ESP32 서버 살아있는지" 확인)
  const connectESP32 = async () => {
    try {
      setStatus("Checking ESP32...");
      const res = await fetch(`${ESP32_BASE_URL}/status`, { method: "GET" });
      if (!res.ok) throw new Error(`ESP32 /status failed: ${res.status}`);

      setConnected(true);
      setStatus("Connected (Wi-Fi)");
    } catch (e: unknown) {
      console.error(e);
      setConnected(false);
      if (e instanceof Error) setStatus(`Error: ${e.message}`);
      else setStatus("Error: unknown");
    }
  };

  // ✅ UI Start = ESP32에게 측정 시작 + 결과 JSON 가져오기
  const measureOnce = async () => {
    if (!connected) {
      setStatus("Not connected. Click Connect first.");
      return;
    }

    try {
      setMeasuring(true);
      setPrediction("");
      setStatus("Starting measure on ESP32...");

      // 1) 측정 시작
      const startRes = await fetch(`${ESP32_BASE_URL}/start`, {
        method: "POST",
      });
      if (!startRes.ok) {
        throw new Error(`ESP32 /start failed: ${startRes.status}`);
      }

      // 2) 결과 가져오기
      // ESP32 구현이 "start 요청이 3초 블로킹 후 완료"라면 아래 GET은 바로 성공함.
      // 혹시 비동기로 바뀌면 폴링 방식으로도 대응 가능하게 폴링을 넣어둠.
      setStatus("Waiting result from ESP32...");
      const deadline = Date.now() + 20000; // 최대 20초 기다림

      while (true) {
        const resultRes = await fetch(`${ESP32_BASE_URL}/result`, {
          method: "GET",
          cache: "no-store",
        });

        if (resultRes.status === 202) {
          // measuring
          if (Date.now() > deadline) throw new Error("ESP32 result timeout");
          await sleep(400);
          continue;
        }

        if (!resultRes.ok) {
          throw new Error(`ESP32 /result failed: ${resultRes.status}`);
        }

        const data = (await resultRes.json()) as SensorData;

        // 샘플 수 검증
        if (
          data.time.length !== TOTAL_SAMPLES ||
          data.value1.length !== TOTAL_SAMPLES ||
          data.value2.length !== TOTAL_SAMPLES
        ) {
          throw new Error(
            `Sample mismatch (got ${data.time.length}, expected ${TOTAL_SAMPLES})`
          );
        }

        setRawData(data);
        setStatus(`Measured (${TOTAL_SAMPLES} samples). Graph updated.`);
        break;
      }
    } catch (e: unknown) {
      console.error(e);
      if (e instanceof Error) setStatus(`Error: ${e.message}`);
      else setStatus("Error: unknown");
    } finally {
      setMeasuring(false);
    }
  };

  // ✅ UI Predict = FastAPI로 rawData 전송해서 예측
  const predictFromLastMeasurement = async () => {
    if (!rawData) {
      setStatus("No data. Please measure first.");
      return;
    }

    try {
      setPredicting(true);
      setStatus("Sending to FastAPI...");

      const res = await fetch(FASTAPI_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(rawData),
      });

      if (!res.ok) {
        throw new Error(`FastAPI error: ${res.status}`);
      }

      const data = (await res.json()) as PredictResponse;
      setPrediction(data.prediction ?? "");
      setStatus("Prediction done");
    } catch (e: unknown) {
      console.error(e);
      if (e instanceof Error) setStatus(`Error: ${e.message}`);
      else setStatus("Error: unknown");
    } finally {
      setPredicting(false);
    }
  };

  return (
    <main className="min-h-screen p-10 bg-gray-50">
      <h1 className="text-3xl font-bold mb-2">ESP32 → LLM Piezo Recognition</h1>
      <p className="text-gray-600 mb-6">
        SAMPLE_RATE={SAMPLE_RATE_HZ}Hz / DURATION={RECORD_DURATION_MS}ms / TOTAL=
        {TOTAL_SAMPLES}
      </p>

      <div className="flex gap-4 mb-4">
        <button
          onClick={connectESP32}
          disabled={connected}
          className={`px-5 py-3 text-white rounded-lg ${
            connected ? "bg-gray-400" : "bg-blue-600"
          }`}
        >
          {connected ? "Connected" : "Connect ESP32 (Wi-Fi)"}
        </button>

        <button
          onClick={measureOnce}
          disabled={!connected || measuring}
          className={`px-5 py-3 text-white rounded-lg ${
            !connected || measuring ? "bg-gray-400" : "bg-indigo-600"
          }`}
        >
          {measuring ? "Measuring..." : "Start / Measure (3s)"}
        </button>

        <button
          onClick={predictFromLastMeasurement}
          disabled={!rawData || predicting}
          className={`px-5 py-3 text-white rounded-lg ${
            !rawData || predicting ? "bg-gray-400" : "bg-green-600"
          }`}
        >
          {predicting ? "Predicting..." : "Predict"}
        </button>
      </div>

      <div className="mb-6 text-sm text-gray-700">
        <span className="font-semibold">Status:</span> {status}
      </div>

      {/* 그래프 */}
      <div className="bg-white p-6 rounded-xl shadow mb-6">
        <h2 className="text-xl font-semibold mb-4">Piezo Sensor Waveform</h2>

        {!rawData ? (
          <p className="text-gray-500">
            아직 데이터 없음. Start/Measure(3s)를 누르면 그래프가 표시됨.
          </p>
        ) : (
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="left" stroke="#2563eb" dot={false} />
                <Line type="monotone" dataKey="right" stroke="#dc2626" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* 예측 결과 */}
      {prediction && (
        <div className="bg-green-50 border border-green-200 p-4 rounded-lg">
          <p className="text-lg font-medium text-green-700">Prediction Result</p>
          <p className="text-2xl font-bold text-green-800 mt-1">{prediction}</p>
        </div>
      )}
    </main>
  );
}
