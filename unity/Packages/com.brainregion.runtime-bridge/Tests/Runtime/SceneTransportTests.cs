using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using NUnit.Framework;

namespace BrainRegion.RuntimeBridge.Tests
{
    public sealed class SceneTransportTests
    {
        private const string ExpectedProof =
            "hmac-sha256.P35gOtwqOOnRuK8qiB0LyozZcDr18lfzgxTs6AC8nj8";

        [Test]
        public void PairingProofMatchesRustGoldenVector()
        {
            Assert.That(ScenePairingProof.TryParseChallenge(
                BuildChallenge("scene.read"),
                1700000000000L,
                out ScenePairingChallenge challenge,
                out string challengeError),
                Is.True,
                challengeError);

            byte[] secret = Encoding.ASCII.GetBytes(
                "0123456789abcdef0123456789abcdef");
            Assert.That(ScenePairingProof.TryCreateRegistration(
                BuildRegistration(
                    SceneCapabilities.SceneRead,
                    SceneCapabilities.SceneWrite,
                    SceneCapabilities.SceneSpawn,
                    SceneCapabilities.SceneUndo,
                    SceneCapabilities.LogsRead),
                challenge,
                secret,
                out string paired,
                out string registrationError),
                Is.True,
                registrationError);
            Assert.That(
                (string)JObject.Parse(paired)["params"]["pairingProof"],
                Is.EqualTo(ExpectedProof));
        }

        [Test]
        public void PairingRejectsExpiredChallengeAndUnadvertisedGrant()
        {
            Assert.That(ScenePairingProof.TryParseChallenge(
                BuildChallenge(SceneCapabilities.SceneRead),
                1800000000000L,
                out _,
                out string expiryError),
                Is.False);
            StringAssert.Contains("expired", expiryError);

            Assert.That(ScenePairingProof.TryParseChallenge(
                BuildChallenge(SceneCapabilities.SceneWrite),
                1700000000000L,
                out ScenePairingChallenge challenge,
                out string challengeError),
                Is.True,
                challengeError);
            Assert.That(ScenePairingProof.TryCreateRegistration(
                BuildRegistration(SceneCapabilities.SceneRead),
                challenge,
                Encoding.ASCII.GetBytes("0123456789abcdef0123456789abcdef"),
                out _,
                out string registrationError),
                Is.False);
            StringAssert.Contains("not advertised", registrationError);
        }

        [Test]
        public async Task JsonLineReaderHandlesCrLfAndRejectsOversizedFrames()
        {
            byte[] input = Encoding.UTF8.GetBytes("test\r\nnext\n");
            using (var stream = new MemoryStream(input))
            {
                var reader = new BoundedJsonLineReader(stream, 4, 128);
                Assert.That(await reader.ReadLineAsync(CancellationToken.None), Is.EqualTo("test"));
                Assert.That(await reader.ReadLineAsync(CancellationToken.None), Is.EqualTo("next"));
                Assert.That(await reader.ReadLineAsync(CancellationToken.None), Is.Null);
            }

            using (var stream = new MemoryStream(Encoding.UTF8.GetBytes("12345\n")))
            {
                var reader = new BoundedJsonLineReader(stream, 4, 128);
                Assert.ThrowsAsync<InvalidDataException>(
                    async () => await reader.ReadLineAsync(CancellationToken.None));
            }
        }

        [Test]
        public void SynchronousJsonLineReaderHandlesCrLfAndRejectsOversizedFrames()
        {
            byte[] input = Encoding.UTF8.GetBytes("test\r\nnext\n");
            using (var stream = new MemoryStream(input))
            {
                var reader = new BoundedJsonLineReader(stream, 4, 128);
                Assert.That(reader.ReadLine(), Is.EqualTo("test"));
                Assert.That(reader.ReadLine(), Is.EqualTo("next"));
                Assert.That(reader.ReadLine(), Is.Null);
            }

            using (var stream = new MemoryStream(Encoding.UTF8.GetBytes("12345\n")))
            {
                var reader = new BoundedJsonLineReader(stream, 4, 128);
                Assert.Throws<InvalidDataException>(() => reader.ReadLine());
            }
        }

        [Test]
        public void PollingJsonLineReaderRetainsPartialAndBufferedFrames()
        {
            byte[] input = Encoding.UTF8.GetBytes("one\ntwo\npartial");
            using (var stream = new MemoryStream(input))
            {
                var reader = new BoundedJsonLineReader(stream, 16, 128);
                Assert.That(reader.TryReadLine(3, out _), Is.False);
                Assert.That(reader.TryReadLine(5, out string first), Is.True);
                Assert.That(first, Is.EqualTo("one"));
                Assert.That(reader.TryReadLine(0, out string second), Is.True);
                Assert.That(second, Is.EqualTo("two"));
                Assert.That(reader.TryReadLine(7, out _), Is.False);
            }
        }

        [Test]
        public async Task WriterQueueBoundsIncludeLineTerminator()
        {
            using (var queue = new BoundedSceneWriterQueue(1, 4))
            {
                Assert.That(queue.TryEnqueue("abc", 3), Is.True);
                Assert.That(queue.TryEnqueue("x", 3), Is.False);
                Assert.That(
                    await queue.DequeueAsync(CancellationToken.None),
                    Is.EqualTo("abc"));
            }
        }

        [Test]
        public void SynchronousWriterQueueBoundsIncludeLineTerminator()
        {
            using (var queue = new BoundedSceneWriterQueue(1, 4))
            {
                Assert.That(queue.TryEnqueue("abc", 3), Is.True);
                Assert.That(queue.TryEnqueue("x", 3), Is.False);
                Assert.That(queue.Dequeue(CancellationToken.None), Is.EqualTo("abc"));
            }
        }

        [Test]
        public void WriterQueueSupportsNonBlockingPoll()
        {
            using (var queue = new BoundedSceneWriterQueue(1, 4))
            {
                Assert.That(queue.TryDequeue(out _), Is.False);
                Assert.That(queue.TryEnqueue("abc", 3), Is.True);
                Assert.That(queue.TryDequeue(out string frame), Is.True);
                Assert.That(frame, Is.EqualTo("abc"));
                Assert.That(queue.TryDequeue(out _), Is.False);
            }
        }

        private static string BuildChallenge(string grant)
        {
            return new JObject
            {
                ["jsonrpc"] = "2.0",
                ["method"] = "runtime/challenge",
                ["params"] = new JObject
                {
                    ["protocolVersion"] = ScenePairingProof.ProtocolVersion,
                    ["algorithm"] = ScenePairingProof.Algorithm,
                    ["nonce"] = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                    ["expiresUnixMs"] = 1800000000000L,
                    ["principalId"] = "unity-local",
                    ["grantedCapabilities"] = new JArray(grant),
                },
            }.ToString(Formatting.None);
        }

        private static string BuildRegistration(params string[] capabilities)
        {
            return new JObject
            {
                ["jsonrpc"] = "2.0",
                ["method"] = "runtime/register",
                ["params"] = new JObject
                {
                    ["protocolVersion"] = SceneProtocol.Version,
                    ["instanceId"] = "player-01",
                    ["sessionId"] = "session-01",
                    ["buildId"] = "windows-il2cpp-dev-001",
                    ["unityVersion"] = "6000.3.20f1",
                    ["platform"] = "WindowsPlayer",
                    ["product"] = "VR Project",
                    ["sceneId"] = "Sandbox",
                    ["sceneRevision"] = 0,
                    ["status"] = "ready",
                    ["error"] = null,
                    ["capabilities"] = new JArray(capabilities),
                    ["pairingProof"] = null,
                },
            }.ToString(Formatting.None);
        }
    }
}
