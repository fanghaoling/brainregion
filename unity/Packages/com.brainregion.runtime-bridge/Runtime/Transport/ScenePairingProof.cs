using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace BrainRegion.RuntimeBridge
{
    public sealed class ScenePairingChallenge
    {
        public string ProtocolVersion { get; }
        public string Algorithm { get; }
        public string Nonce { get; }
        public long ExpiresUnixMs { get; }
        public string PrincipalId { get; }
        public IReadOnlyList<string> GrantedCapabilities { get; }

        internal ScenePairingChallenge(
            string protocolVersion,
            string algorithm,
            string nonce,
            long expiresUnixMs,
            string principalId,
            IList<string> grantedCapabilities)
        {
            ProtocolVersion = protocolVersion;
            Algorithm = algorithm;
            Nonce = nonce;
            ExpiresUnixMs = expiresUnixMs;
            PrincipalId = principalId;
            GrantedCapabilities = new ReadOnlyCollection<string>(grantedCapabilities);
        }
    }

    /// <summary>
    /// Strict parser and cross-language HMAC proof builder for
    /// brainregion.scene.pairing.v1. It never stores the pre-shared secret.
    /// </summary>
    public static class ScenePairingProof
    {
        public const string ProtocolVersion = "brainregion.scene.pairing.v1";
        public const string Algorithm = "hmac-sha256";
        private const string ProofPrefix = "hmac-sha256.";
        private const int MinimumSecretBytes = 32;
        private const long MaximumJsonSafeInteger = 9007199254740991L;

        private static readonly HashSet<string> ChallengeEnvelopeFields =
            new HashSet<string>(new[] { "jsonrpc", "method", "params" }, StringComparer.Ordinal);
        private static readonly HashSet<string> ChallengeFields =
            new HashSet<string>(new[]
            {
                "protocolVersion", "algorithm", "nonce", "expiresUnixMs", "principalId",
                "grantedCapabilities",
            }, StringComparer.Ordinal);
        private static readonly HashSet<string> RegistrationEnvelopeFields =
            new HashSet<string>(new[] { "jsonrpc", "method", "params" }, StringComparer.Ordinal);
        private static readonly HashSet<string> RegistrationFields =
            new HashSet<string>(new[]
            {
                "protocolVersion", "instanceId", "sessionId", "buildId", "unityVersion",
                "platform", "product", "sceneId", "sceneRevision", "status", "error",
                "capabilities", "pairingProof",
            }, StringComparer.Ordinal);

        public static bool TryParseChallenge(
            string json,
            out ScenePairingChallenge challenge,
            out string error)
        {
            return TryParseChallenge(
                json,
                DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
                out challenge,
                out error);
        }

        internal static bool TryParseChallenge(
            string json,
            long nowUnixMs,
            out ScenePairingChallenge challenge,
            out string error)
        {
            challenge = null;
            error = null;
            try
            {
                JObject envelope = JObject.Parse(json);
                RequireExactFields(envelope, ChallengeEnvelopeFields, "challenge envelope");
                RequireString(envelope, "jsonrpc", "2.0");
                RequireString(envelope, "method", "runtime/challenge");
                JObject parameters = RequireObject(envelope, "params");
                RequireExactFields(parameters, ChallengeFields, "challenge params");

                string protocolVersion = RequireString(parameters, "protocolVersion");
                string algorithm = RequireString(parameters, "algorithm");
                string nonce = RequireString(parameters, "nonce");
                long expiresUnixMs = RequireInteger(parameters, "expiresUnixMs");
                string principalId = RequireString(parameters, "principalId");
                List<string> grants = RequireCapabilities(parameters, "grantedCapabilities");

                if (!string.Equals(protocolVersion, ProtocolVersion, StringComparison.Ordinal))
                    throw new FormatException("Unsupported pairing protocol version");
                if (!string.Equals(algorithm, Algorithm, StringComparison.Ordinal))
                    throw new FormatException("Unsupported pairing algorithm");
                if (!SceneProtocol.IsIdentifier(principalId, 128))
                    throw new FormatException("Pairing principalId is invalid");
                byte[] nonceBytes = DecodeBase64Url(nonce);
                if (nonceBytes.Length != 32)
                    throw new FormatException("Pairing nonce must contain 32 bytes");
                Array.Clear(nonceBytes, 0, nonceBytes.Length);
                if (expiresUnixMs < 0 || expiresUnixMs > MaximumJsonSafeInteger)
                    throw new FormatException("Pairing challenge expiry is outside the JSON safe range");
                if (expiresUnixMs <= nowUnixMs)
                    throw new FormatException("Pairing challenge has expired");

                challenge = new ScenePairingChallenge(
                    protocolVersion,
                    algorithm,
                    nonce,
                    expiresUnixMs,
                    principalId,
                    grants);
                return true;
            }
            catch (Exception exception) when (
                exception is JsonException || exception is FormatException ||
                exception is OverflowException || exception is ArgumentException)
            {
                error = exception.Message;
                return false;
            }
        }

        public static bool TryCreateRegistration(
            string baseRegistrationJson,
            ScenePairingChallenge challenge,
            byte[] secret,
            out string pairedRegistrationJson,
            out string error)
        {
            pairedRegistrationJson = null;
            error = null;
            try
            {
                if (challenge == null) throw new ArgumentNullException(nameof(challenge));
                if (secret == null || secret.Length < MinimumSecretBytes)
                    throw new ArgumentException("Pairing secret must contain at least 32 bytes");

                JObject envelope = JObject.Parse(baseRegistrationJson);
                RequireExactFields(envelope, RegistrationEnvelopeFields, "registration envelope");
                RequireString(envelope, "jsonrpc", "2.0");
                RequireString(envelope, "method", "runtime/register");
                JObject registration = RequireObject(envelope, "params");
                RequireAllowedFields(registration, RegistrationFields, "registration params");

                string protocolVersion = RequireString(registration, "protocolVersion");
                if (!string.Equals(protocolVersion, SceneProtocol.Version, StringComparison.Ordinal))
                    throw new FormatException("Registration protocolVersion is invalid");
                string instanceId = RequireIdentifier(registration, "instanceId", 128);
                string sessionId = RequireIdentifier(registration, "sessionId", 128);
                string buildId = RequireBoundedString(registration, "buildId", 256);
                string unityVersion = RequireBoundedString(registration, "unityVersion", 64);
                string platform = RequireBoundedString(registration, "platform", 64);
                string product = RequireBoundedString(registration, "product", 256);
                string sceneId = RequireBoundedString(registration, "sceneId", 256);
                long sceneRevision = RequireInteger(registration, "sceneRevision");
                if (sceneRevision < 0 || sceneRevision > 9007199254740991L)
                    throw new FormatException("Registration sceneRevision is outside the safe range");
                string status = RequireString(registration, "status");
                if (status != "ready" && status != "degraded")
                    throw new FormatException("Registration status is invalid");
                string runtimeError = RequireOptionalString(registration, "error", 4096);
                List<string> capabilities = RequireCapabilities(registration, "capabilities");
                if (capabilities.Count == 0)
                    throw new FormatException("Registration must advertise at least one capability");
                foreach (string grant in challenge.GrantedCapabilities)
                {
                    if (!capabilities.Contains(grant))
                        throw new FormatException("Server granted a capability not advertised by this Player");
                }

                byte[] payload = BuildCanonicalPayload(
                    challenge,
                    protocolVersion,
                    instanceId,
                    sessionId,
                    buildId,
                    unityVersion,
                    platform,
                    product,
                    sceneId,
                    sceneRevision,
                    status,
                    runtimeError,
                    capabilities);
                byte[] tag;
                using (var hmac = new HMACSHA256(secret))
                    tag = hmac.ComputeHash(payload);
                Array.Clear(payload, 0, payload.Length);
                string proof = ProofPrefix + EncodeBase64Url(tag);
                Array.Clear(tag, 0, tag.Length);

                registration["pairingProof"] = proof;
                pairedRegistrationJson = envelope.ToString(Formatting.None);
                return true;
            }
            catch (Exception exception) when (
                exception is JsonException || exception is FormatException ||
                exception is OverflowException || exception is ArgumentException ||
                exception is CryptographicException)
            {
                error = exception.Message;
                return false;
            }
        }

        private static byte[] BuildCanonicalPayload(
            ScenePairingChallenge challenge,
            string protocolVersion,
            string instanceId,
            string sessionId,
            string buildId,
            string unityVersion,
            string platform,
            string product,
            string sceneId,
            long sceneRevision,
            string status,
            string runtimeError,
            IList<string> capabilities)
        {
            using (var output = new MemoryStream(512))
            {
                AppendField(output, challenge.ProtocolVersion);
                AppendField(output, challenge.Algorithm);
                AppendField(output, challenge.Nonce);
                AppendField(output, challenge.ExpiresUnixMs.ToString(CultureInfo.InvariantCulture));
                AppendField(output, challenge.PrincipalId);
                List<string> grants = new List<string>(challenge.GrantedCapabilities);
                grants.Sort(StringComparer.Ordinal);
                AppendField(output, grants.Count.ToString(CultureInfo.InvariantCulture));
                foreach (string grant in grants) AppendField(output, grant);
                AppendField(output, protocolVersion);
                AppendField(output, instanceId);
                AppendField(output, sessionId);
                AppendField(output, buildId);
                AppendField(output, unityVersion);
                AppendField(output, platform);
                AppendField(output, product);
                AppendField(output, sceneId);
                AppendField(output, sceneRevision.ToString(CultureInfo.InvariantCulture));
                AppendField(output, status);
                if (runtimeError == null)
                {
                    output.WriteByte(0);
                }
                else
                {
                    output.WriteByte(1);
                    AppendField(output, runtimeError);
                }
                List<string> orderedCapabilities = new List<string>(capabilities);
                orderedCapabilities.Sort(StringComparer.Ordinal);
                AppendField(
                    output,
                    orderedCapabilities.Count.ToString(CultureInfo.InvariantCulture));
                foreach (string capability in orderedCapabilities) AppendField(output, capability);
                return output.ToArray();
            }
        }

        private static void AppendField(Stream output, string value)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(value);
            byte[] length = Encoding.ASCII.GetBytes(bytes.Length.ToString(CultureInfo.InvariantCulture));
            output.Write(length, 0, length.Length);
            output.WriteByte((byte)':');
            output.Write(bytes, 0, bytes.Length);
            output.WriteByte((byte)'\n');
        }

        private static string RequireIdentifier(JObject value, string name, int maximum)
        {
            string result = RequireString(value, name);
            if (!SceneProtocol.IsIdentifier(result, maximum))
                throw new FormatException($"{name} is not a valid protocol identifier");
            return result;
        }

        private static string RequireBoundedString(JObject value, string name, int maximum)
        {
            string result = RequireString(value, name);
            if (result.Length == 0 || Encoding.UTF8.GetByteCount(result) > maximum)
                throw new FormatException($"{name} must contain 1..{maximum} UTF-8 bytes");
            return result;
        }

        private static string RequireOptionalString(JObject value, string name, int maximum)
        {
            if (!value.TryGetValue(name, StringComparison.Ordinal, out JToken token))
                throw new FormatException($"{name} is required");
            if (token.Type == JTokenType.Null) return null;
            if (token.Type != JTokenType.String)
                throw new FormatException($"{name} must be a string or null");
            string result = token.Value<string>();
            if (Encoding.UTF8.GetByteCount(result) > maximum)
                throw new FormatException($"{name} exceeds {maximum} UTF-8 bytes");
            return result;
        }

        private static long RequireInteger(JObject value, string name)
        {
            if (!value.TryGetValue(name, StringComparison.Ordinal, out JToken token) ||
                token.Type != JTokenType.Integer)
                throw new FormatException($"{name} must be an integer");
            return token.Value<long>();
        }

        private static List<string> RequireCapabilities(JObject value, string name)
        {
            if (!value.TryGetValue(name, StringComparison.Ordinal, out JToken token) ||
                !(token is JArray array))
                throw new FormatException($"{name} must be an array");
            var result = new List<string>(array.Count);
            var unique = new HashSet<string>(StringComparer.Ordinal);
            foreach (JToken item in array)
            {
                if (item.Type != JTokenType.String)
                    throw new FormatException($"{name} entries must be strings");
                string capability = item.Value<string>();
                if (!SceneCapabilities.IsKnown(capability) || !unique.Add(capability))
                    throw new FormatException($"{name} contains an unknown or duplicate capability");
                result.Add(capability);
            }
            result.Sort(StringComparer.Ordinal);
            return result;
        }

        private static JObject RequireObject(JObject value, string name)
        {
            if (!value.TryGetValue(name, StringComparison.Ordinal, out JToken token) ||
                !(token is JObject result))
                throw new FormatException($"{name} must be an object");
            return result;
        }

        private static string RequireString(JObject value, string name, string expected = null)
        {
            if (!value.TryGetValue(name, StringComparison.Ordinal, out JToken token) ||
                token.Type != JTokenType.String)
                throw new FormatException($"{name} must be a string");
            string result = token.Value<string>();
            if (expected != null && !string.Equals(result, expected, StringComparison.Ordinal))
                throw new FormatException($"{name} has an unsupported value");
            return result;
        }

        private static void RequireExactFields(
            JObject value,
            ISet<string> expected,
            string context)
        {
            RequireAllowedFields(value, expected, context);
            foreach (string field in expected)
            {
                if (value.Property(field, StringComparison.Ordinal) == null)
                    throw new FormatException($"{context} is missing {field}");
            }
        }

        private static void RequireAllowedFields(
            JObject value,
            ISet<string> allowed,
            string context)
        {
            JProperty unexpected = value.Properties().FirstOrDefault(
                property => !allowed.Contains(property.Name));
            if (unexpected != null)
                throw new FormatException($"{context} contains unexpected field {unexpected.Name}");
        }

        private static string EncodeBase64Url(byte[] value)
        {
            return Convert.ToBase64String(value)
                .TrimEnd('=')
                .Replace('+', '-')
                .Replace('/', '_');
        }

        private static byte[] DecodeBase64Url(string value)
        {
            if (string.IsNullOrEmpty(value) || value.Any(character =>
                    !(char.IsLetterOrDigit(character) || character == '-' || character == '_')))
                throw new FormatException("Pairing nonce is not base64url");
            string padded = value.Replace('-', '+').Replace('_', '/');
            switch (padded.Length % 4)
            {
                case 2: padded += "=="; break;
                case 3: padded += "="; break;
                case 0: break;
                default: throw new FormatException("Pairing nonce has invalid base64url length");
            }
            return Convert.FromBase64String(padded);
        }
    }
}
