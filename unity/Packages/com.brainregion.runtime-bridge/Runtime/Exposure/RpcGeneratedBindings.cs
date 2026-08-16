using System;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace BrainRegion.RuntimeBridge
{
    /// <summary>
    /// Marks a public MonoBehaviour as a source for an AOT-safe generated Scene RPC
    /// property adapter. Reflection is used by the Unity Editor generator only;
    /// generated Player code accesses members directly.
    /// </summary>
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = false)]
    public sealed class RpcBindingTargetAttribute : Attribute
    {
        public RpcBindingTargetAttribute(string componentKey, string typeId)
        {
            ComponentKey = componentKey;
            TypeId = typeId;
        }

        public string ComponentKey { get; }
        public string TypeId { get; }
    }

    /// <summary>
    /// Exposes a public writable field or property through generated bindings.
    /// v1 generation supports bool, int, float, double, and string members.
    /// </summary>
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property,
        AllowMultiple = false, Inherited = true)]
    public sealed class RpcExposedPropertyAttribute : Attribute
    {
        public RpcExposedPropertyAttribute(string propertyId)
        {
            PropertyId = propertyId;
        }

        public string PropertyId { get; }
        public string DisplayName { get; set; }
        public bool Persistent { get; set; } = true;
        public double Minimum { get; set; } = double.NaN;
        public double Maximum { get; set; } = double.NaN;
        public int MaximumLength { get; set; } = 16384;
    }

    /// <summary>
    /// Shared non-reflective validators called by generated adapters. Every helper
    /// is exception-safe so project-controlled wire values cannot escape the adapter
    /// contract through numeric conversion errors.
    /// </summary>
    public static class RpcGeneratedBindingValues
    {
        public static bool TryBoolean(
            JToken proposed,
            out JToken canonical,
            out string error)
        {
            if (proposed == null || proposed.Type != JTokenType.Boolean)
            {
                canonical = null;
                error = "value must be a boolean";
                return false;
            }
            canonical = new JValue(proposed.Value<bool>());
            error = null;
            return true;
        }

        public static bool TryInteger(
            JToken proposed,
            double? minimum,
            double? maximum,
            out JToken canonical,
            out string error)
        {
            canonical = null;
            if (proposed == null || proposed.Type != JTokenType.Integer)
            {
                error = "value must be an integer";
                return false;
            }
            try
            {
                long candidate = proposed.Value<long>();
                if (candidate < int.MinValue || candidate > int.MaxValue ||
                    minimum.HasValue && candidate < minimum.Value ||
                    maximum.HasValue && candidate > maximum.Value)
                {
                    error = "integer value is outside the exposed range";
                    return false;
                }
                canonical = new JValue((int)candidate);
                error = null;
                return true;
            }
            catch (Exception exception) when (
                exception is FormatException || exception is InvalidCastException ||
                exception is OverflowException)
            {
                error = "integer value could not be converted safely";
                return false;
            }
        }

        public static bool TryNumber(
            JToken proposed,
            double? minimum,
            double? maximum,
            bool singlePrecision,
            out JToken canonical,
            out string error)
        {
            canonical = null;
            if (proposed == null ||
                proposed.Type != JTokenType.Integer && proposed.Type != JTokenType.Float)
            {
                error = "value must be a finite number";
                return false;
            }
            try
            {
                double candidate = proposed.Value<double>();
                if (double.IsNaN(candidate) || double.IsInfinity(candidate) ||
                    singlePrecision && (candidate < -float.MaxValue || candidate > float.MaxValue) ||
                    minimum.HasValue && candidate < minimum.Value ||
                    maximum.HasValue && candidate > maximum.Value)
                {
                    error = "number is outside the exposed range";
                    return false;
                }
                canonical = singlePrecision
                    ? new JValue((float)candidate)
                    : new JValue(candidate);
                error = null;
                return true;
            }
            catch (Exception exception) when (
                exception is FormatException || exception is InvalidCastException ||
                exception is OverflowException)
            {
                error = "number could not be converted safely";
                return false;
            }
        }

        public static bool TryString(
            JToken proposed,
            int maximumLength,
            out JToken canonical,
            out string error)
        {
            if (proposed == null || proposed.Type != JTokenType.String)
            {
                canonical = null;
                error = "value must be a string";
                return false;
            }
            string candidate = proposed.Value<string>();
            if (candidate == null || candidate.Length > maximumLength)
            {
                canonical = null;
                error = "string exceeds the exposed length limit";
                return false;
            }
            canonical = new JValue(candidate);
            error = null;
            return true;
        }
    }
}
