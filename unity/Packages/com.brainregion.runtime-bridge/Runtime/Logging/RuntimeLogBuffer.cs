using System;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace BrainRegion.RuntimeBridge
{
    /// <summary>
    /// Bounded log capture safe for Application.logMessageReceivedThreaded.
    /// The callback only copies supplied values under a lock and never calls Unity APIs.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class RuntimeLogBuffer : MonoBehaviour
    {
        [SerializeField, Min(16)] private int capacity = 1000;

        private readonly object gate = new object();
        private readonly LinkedList<Entry> entries = new LinkedList<Entry>();
        private long nextSequence;

        private void OnEnable()
        {
            Application.logMessageReceivedThreaded += OnLogMessage;
        }

        private void OnDisable()
        {
            Application.logMessageReceivedThreaded -= OnLogMessage;
        }

        public JObject Poll(LogsPollRequest request)
        {
            request = request ?? new LogsPollRequest();
            int limit = Math.Min(
                SceneProtocol.MaxPageSize,
                Math.Max(1, request.Limit <= 0 ? 200 : request.Limit));
            HashSet<string> levels = request.Levels == null
                ? null
                : new HashSet<string>(request.Levels, StringComparer.OrdinalIgnoreCase);

            lock (gate)
            {
                long oldest = entries.First?.Value.Sequence ?? nextSequence + 1;
                var serialized = new JArray();
                long next = request.AfterSeq;
                foreach (Entry entry in entries)
                {
                    if (entry.Sequence <= request.AfterSeq) continue;
                    if (levels != null && !levels.Contains(entry.Level)) continue;
                    serialized.Add(new JObject
                    {
                        ["seq"] = entry.Sequence,
                        ["timestampUnixMs"] = entry.TimestampUnixMs,
                        ["level"] = entry.Level,
                        ["message"] = entry.Message,
                        ["stackTrace"] = entry.StackTrace,
                    });
                    next = entry.Sequence;
                    if (serialized.Count >= limit) break;
                }

                return new JObject
                {
                    ["entries"] = serialized,
                    ["nextSeq"] = next,
                    ["droppedBefore"] = request.AfterSeq < oldest - 1 ? oldest : 0,
                };
            }
        }

        private void OnLogMessage(string message, string stackTrace, LogType type)
        {
            string level = ToLevel(type);
            var entry = new Entry
            {
                TimestampUnixMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
                Level = level,
                Message = Truncate(message, 16384),
                StackTrace = Truncate(stackTrace, 65536),
            };

            lock (gate)
            {
                entry.Sequence = ++nextSequence;
                entries.AddLast(entry);
                int max = Math.Max(16, capacity);
                while (entries.Count > max)
                    entries.RemoveFirst();
            }
        }

        private static string ToLevel(LogType type)
        {
            switch (type)
            {
                case LogType.Warning: return "warning";
                case LogType.Error: return "error";
                case LogType.Exception: return "exception";
                case LogType.Assert: return "assert";
                default: return "info";
            }
        }

        private static string Truncate(string value, int maximum)
        {
            if (string.IsNullOrEmpty(value) || value.Length <= maximum) return value ?? string.Empty;
            return value.Substring(0, maximum);
        }

        private sealed class Entry
        {
            public long Sequence;
            public long TimestampUnixMs;
            public string Level;
            public string Message;
            public string StackTrace;
        }
    }
}
