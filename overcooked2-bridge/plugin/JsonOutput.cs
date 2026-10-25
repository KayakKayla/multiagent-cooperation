using System;
using System.Collections;
using System.Globalization;
using System.Reflection;
using System.Text;

namespace Sims.Overcooked2Bridge
{
    // Unity 2017's native serializer omitted nested DTOs from this loaded mod.
    // Serialize our small public-field DTO tree entirely in managed code instead.
    public static class JsonOutput
    {
        public static string Write(object value)
        {
            var output = new StringBuilder();
            Append(output, value);
            return output.ToString();
        }
        static void Append(StringBuilder output, object value)
        {
            if (value == null) { output.Append("null"); return; }
            if (value is string) { Quote(output, (string)value); return; }
            if (value is bool) { output.Append((bool)value ? "true" : "false"); return; }
            if (value is float || value is double)
            {
                double number = Convert.ToDouble(value, CultureInfo.InvariantCulture);
                if (double.IsNaN(number) || double.IsInfinity(number)) throw new ArgumentException("Nonfinite state value");
                output.Append(((IFormattable)value).ToString("R", CultureInfo.InvariantCulture)); return;
            }
            if (value is int || value is long)
            { output.Append(Convert.ToString(value, CultureInfo.InvariantCulture)); return; }
            var sequence = value as IEnumerable;
            if (sequence != null)
            {
                output.Append('['); bool first = true;
                foreach (object element in sequence)
                {
                    if (!first) output.Append(','); first = false;
                    Append(output, element);
                }
                output.Append(']'); return;
            }
            output.Append('{'); bool firstField = true;
            foreach (FieldInfo field in value.GetType().GetFields(BindingFlags.Instance | BindingFlags.Public))
            {
                if (!firstField) output.Append(','); firstField = false;
                Quote(output, field.Name); output.Append(':'); Append(output, field.GetValue(value));
            }
            output.Append('}');
        }
        static void Quote(StringBuilder output, string value)
        {
            output.Append('"');
            foreach (char ch in value)
            {
                if (ch == '"' || ch == '\\') { output.Append('\\'); output.Append(ch); }
                else if (ch < 32 || char.IsSurrogate(ch)) output.Append("\\u" + ((int)ch).ToString("x4", CultureInfo.InvariantCulture));
                else output.Append(ch);
            }
            output.Append('"');
        }
    }
}
