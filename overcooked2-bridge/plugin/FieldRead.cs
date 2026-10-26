using System;
using System.Collections.Generic;
using System.Reflection;

namespace Sims.Overcooked2Bridge
{
    // Plain reflection only; shared with tests that have no Unity dependency.
    public static class FieldRead
    {
        static readonly Dictionary<string, FieldInfo> fields = new Dictionary<string, FieldInfo>();
        public static T Value<T>(object source, string name)
        {
            if (source == null) throw new InvalidOperationException("Uninitialized " + name);
            Type type = source.GetType();
            string key = type.FullName + ":" + name;
            FieldInfo field;
            if (!fields.TryGetValue(key, out field))
            {
                for (Type current = type; current != null && field == null; current = current.BaseType)
                    field = current.GetField(name, BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.DeclaredOnly);
                if (field == null) throw new MissingFieldException(type.FullName, name);
                fields[key] = field;
            }
            return (T)field.GetValue(source);
        }
        public static object ArrayEntry(object source, string name, int index)
        { return Value<Array>(source, name).GetValue(index); }
    }
}
