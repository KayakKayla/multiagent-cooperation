using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace Sims.Overcooked2Bridge
{
    // The wire protocol intentionally accepts only flat command objects.
    public sealed class Command
    {
        public string Op, RequestId, SceneToken;
        public int ChefId, DurationMs;
        public float X, Y;

        public static Command Parse(string json)
        {
            var fields = new FlatObject(json).Read();
            foreach (string key in fields.Keys)
                if (Array.IndexOf(new[] { "op", "request_id", "scene_token", "chef_id", "x", "y", "duration_ms" }, key) < 0)
                    throw new ArgumentException("Unknown field: " + key);
            var c = new Command { Op = Text(fields, "op"), RequestId = Text(fields, "request_id", false) };
            if (c.RequestId.Length > 64) throw new ArgumentException("request_id is too long");
            if (c.Op == "state" || c.Op == "geometry" || c.Op == "release") return c;
            if (c.Op != "move" && c.Op != "pickup") throw new ArgumentException("Unknown operation");
            c.SceneToken = Text(fields, "scene_token");
            c.ChefId = Integer(fields, "chef_id");
            if (c.ChefId == 0) throw new ArgumentException("chef_id must identify a listed chef");
            if (c.Op == "move")
            {
                c.DurationMs = Integer(fields, "duration_ms");
                c.X = Number(fields, "x"); c.Y = Number(fields, "y");
                if (c.DurationMs < 1 || c.DurationMs > 500)
                    throw new ArgumentException("duration_ms must be between 1 and 500");
                float length = c.X * c.X + c.Y * c.Y;
                if (length <= 0 || length > 1.00001f)
                    throw new ArgumentException("Movement vector length must be greater than 0 and at most 1");
            }
            return c;
        }

        static string Text(Dictionary<string, object> fields, string key, bool required = true)
        {
            object value;
            if (!fields.TryGetValue(key, out value) && !required) return "";
            if (!(value is string) || (required && ((string)value).Length == 0))
                throw new ArgumentException(key + " must be a nonempty string");
            return (string)value;
        }
        static int Integer(Dictionary<string, object> fields, string key)
        {
            object value; int result;
            if (!fields.TryGetValue(key, out value) || !(value is JsonNumber) ||
                !int.TryParse(((JsonNumber)value).Value, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out result))
                throw new ArgumentException(key + " must be an integer");
            return result;
        }
        static float Number(Dictionary<string, object> fields, string key)
        {
            object value; float result;
            if (!fields.TryGetValue(key, out value) || !(value is JsonNumber) ||
                !float.TryParse(((JsonNumber)value).Value, NumberStyles.Float, CultureInfo.InvariantCulture, out result) ||
                float.IsNaN(result) || float.IsInfinity(result))
                throw new ArgumentException(key + " must be a finite number");
            return result;
        }
    }

    sealed class JsonNumber
    {
        public string Value;
        public JsonNumber(string value) { Value = value; }
    }

    sealed class FlatObject
    {
        readonly string text;
        int cursor;
        public FlatObject(string text) { this.text = text; }
        void White() { while (cursor < text.Length && " \r\n\t".IndexOf(text[cursor]) >= 0) cursor++; }
        bool Take(char ch) { White(); if (cursor < text.Length && text[cursor] == ch) { cursor++; return true; } return false; }
        void Need(char ch) { if (!Take(ch)) throw new ArgumentException("Invalid JSON: expected " + ch); }
        public Dictionary<string, object> Read()
        {
            var result = new Dictionary<string, object>();
            Need('{');
            if (!Take('}'))
            {
                do
                {
                    string key = String(); Need(':'); White();
                    if (cursor >= text.Length) throw new ArgumentException("Incomplete JSON");
                    object value;
                    if (text[cursor] == '"') value = String();
                    else value = Numeric();
                    if (result.ContainsKey(key)) throw new ArgumentException("Duplicate JSON field: " + key);
                    result.Add(key, value);
                } while (Take(','));
                Need('}');
            }
            White();
            if (cursor != text.Length) throw new ArgumentException("Trailing JSON data");
            return result;
        }
        string String()
        {
            Need('"'); var result = new StringBuilder();
            while (cursor < text.Length)
            {
                char ch = text[cursor++];
                if (ch == '"') return result.ToString();
                if (ch < 32) throw new ArgumentException("Control character in JSON string");
                if (ch == '\\')
                {
                    if (cursor >= text.Length) break;
                    ch = text[cursor++];
                    switch (ch)
                    {
                        case '"': case '\\': case '/': break;
                        case 'b': ch = '\b'; break;
                        case 'f': ch = '\f'; break;
                        case 'n': ch = '\n'; break;
                        case 'r': ch = '\r'; break;
                        case 't': ch = '\t'; break;
                        case 'u':
                            ushort code;
                            if (cursor + 4 > text.Length || !ushort.TryParse(text.Substring(cursor, 4), NumberStyles.HexNumber, CultureInfo.InvariantCulture, out code))
                                throw new ArgumentException("Invalid Unicode escape");
                            cursor += 4; ch = (char)code; break;
                        default: throw new ArgumentException("Invalid JSON escape");
                    }
                }
                result.Append(ch);
            }
            throw new ArgumentException("Unclosed JSON string");
        }
        JsonNumber Numeric()
        {
            int start = cursor;
            if (cursor < text.Length && text[cursor] == '-') cursor++;
            if (cursor < text.Length && text[cursor] == '0') cursor++;
            else Digits();
            if (cursor < text.Length && text[cursor] == '.') { cursor++; Digits(); }
            if (cursor < text.Length && (text[cursor] == 'e' || text[cursor] == 'E'))
            {
                cursor++;
                if (cursor < text.Length && (text[cursor] == '+' || text[cursor] == '-')) cursor++;
                Digits();
            }
            return new JsonNumber(text.Substring(start, cursor - start));
        }
        void Digits()
        {
            int start = cursor;
            while (cursor < text.Length && text[cursor] >= '0' && text[cursor] <= '9') cursor++;
            if (cursor == start) throw new ArgumentException("Invalid JSON number");
        }
    }

    // No Unity dependencies: the same deadline/disconnect gate is exercised by tests.
    public sealed class ActionWindow
    {
        readonly Func<double> clock;
        readonly Func<bool> connected;
        readonly double deadline;
        public bool Released;
        public ActionWindow(Func<double> clock, Func<bool> connected, int durationMs)
        {
            this.clock = clock; this.connected = connected;
            deadline = clock() + durationMs / 1000.0;
        }
        public bool Live { get { return !Released && connected() && clock() < deadline; } }
    }
}
