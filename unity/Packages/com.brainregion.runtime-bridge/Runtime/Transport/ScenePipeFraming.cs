using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace BrainRegion.RuntimeBridge
{
    internal sealed class BoundedJsonLineReader
    {
        private static readonly UTF8Encoding StrictUtf8 = new UTF8Encoding(false, true);
        private readonly Stream stream;
        private readonly int maximumBytes;
        private readonly byte[] readBuffer;
        private int readOffset;
        private int readCount;

        public BoundedJsonLineReader(Stream stream, int maximumBytes, int bufferBytes = 8192)
        {
            this.stream = stream ?? throw new ArgumentNullException(nameof(stream));
            if (maximumBytes < 1) throw new ArgumentOutOfRangeException(nameof(maximumBytes));
            if (bufferBytes < 128) throw new ArgumentOutOfRangeException(nameof(bufferBytes));
            this.maximumBytes = maximumBytes;
            readBuffer = new byte[bufferBytes];
        }

        public async Task<string> ReadLineAsync(CancellationToken cancellationToken)
        {
            using (var line = new MemoryStream())
            {
                while (true)
                {
                    int newline = FindNewline();
                    if (newline >= 0)
                    {
                        int length = newline - readOffset;
                        if (length > 0) line.Write(readBuffer, readOffset, length);
                        readOffset = newline + 1;
                        readCount -= length + 1;
                        return DecodeLine(line);
                    }

                    if (readCount > 0)
                    {
                        line.Write(readBuffer, readOffset, readCount);
                        readOffset = 0;
                        readCount = 0;
                        EnsureBounded(line.Length);
                    }

                    int received = await stream.ReadAsync(
                        readBuffer,
                        0,
                        readBuffer.Length,
                        cancellationToken).ConfigureAwait(false);
                    if (received == 0)
                    {
                        if (line.Length == 0) return null;
                        return DecodeLine(line);
                    }
                    readOffset = 0;
                    readCount = received;
                }
            }
        }

        private int FindNewline()
        {
            int end = readOffset + readCount;
            for (int index = readOffset; index < end; index++)
            {
                if (readBuffer[index] == (byte)'\n') return index;
            }
            return -1;
        }

        private string DecodeLine(MemoryStream line)
        {
            if (line.Length > 0)
            {
                byte[] raw = line.GetBuffer();
                if (raw[line.Length - 1] == (byte)'\r') line.SetLength(line.Length - 1);
            }
            if (line.Length > maximumBytes)
                throw new InvalidDataException($"Scene RPC frame exceeds {maximumBytes} bytes");
            return StrictUtf8.GetString(line.GetBuffer(), 0, checked((int)line.Length));
        }

        private void EnsureBounded(long length)
        {
            // Permit one trailing CR until the line terminator is observed.
            if (length > maximumBytes + 1L)
                throw new InvalidDataException($"Scene RPC frame exceeds {maximumBytes} bytes");
        }
    }

    internal sealed class BoundedSceneWriterQueue : IDisposable
    {
        private readonly object gate = new object();
        private readonly Queue<Frame> frames = new Queue<Frame>();
        private readonly SemaphoreSlim available = new SemaphoreSlim(0);
        private readonly int maximumFrames;
        private readonly int maximumBytes;
        private int queuedBytes;
        private bool disposed;

        public BoundedSceneWriterQueue(int maximumFrames, int maximumBytes)
        {
            if (maximumFrames < 1) throw new ArgumentOutOfRangeException(nameof(maximumFrames));
            if (maximumBytes < 1) throw new ArgumentOutOfRangeException(nameof(maximumBytes));
            this.maximumFrames = maximumFrames;
            this.maximumBytes = maximumBytes;
        }

        public bool TryEnqueue(string json, int frameLimitBytes)
        {
            if (json == null) return false;
            int payloadBytes = Encoding.UTF8.GetByteCount(json);
            if (payloadBytes > frameLimitBytes) return false;
            int framedBytes = checked(payloadBytes + 1);
            lock (gate)
            {
                if (disposed || frames.Count >= maximumFrames ||
                    queuedBytes > maximumBytes - framedBytes)
                    return false;
                frames.Enqueue(new Frame(json, framedBytes));
                queuedBytes += framedBytes;
                available.Release();
            }
            return true;
        }

        public async Task<string> DequeueAsync(CancellationToken cancellationToken)
        {
            await available.WaitAsync(cancellationToken).ConfigureAwait(false);
            lock (gate)
            {
                if (disposed) throw new ObjectDisposedException(nameof(BoundedSceneWriterQueue));
                Frame frame = frames.Dequeue();
                queuedBytes -= frame.Bytes;
                return frame.Json;
            }
        }

        public void Dispose()
        {
            lock (gate)
            {
                if (disposed) return;
                disposed = true;
                frames.Clear();
                queuedBytes = 0;
            }
            available.Dispose();
        }

        private readonly struct Frame
        {
            public readonly string Json;
            public readonly int Bytes;

            public Frame(string json, int bytes)
            {
                Json = json;
                Bytes = bytes;
            }
        }
    }
}
