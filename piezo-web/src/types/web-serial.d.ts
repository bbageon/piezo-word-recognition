// src/types/web-serial.d.ts

interface SerialPort {
  writable: WritableStream<Uint8Array> | null
  open(options: { baudRate: number }): Promise<void>;
  readable: ReadableStream<Uint8Array> | null;
}

interface Serial {
  requestPort(): Promise<SerialPort>;
}

interface Navigator {
  serial: Serial;
}

