using System;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using Sims.Overcooked2Bridge;

class CoreTests
{
    public class Nested { public string name = "chef\n\"1\""; public int[] ids = { 1, -2 }; }
    public class StateReply { public bool ok = true; public Nested state = new Nested(); public string before = null; }
    static int checks;
    static void Check(bool condition, string message)
    {
        if (!condition) throw new Exception(message);
        checks++;
    }
    static void Reject(string json)
    {
        try { Command.Parse(json); }
        catch (ArgumentException) { checks++; return; }
        throw new Exception("Accepted invalid request: " + json);
    }
    static string Move(string duration, string x = "1", string y = "0")
    {
        return "{\"op\":\"move\",\"scene_token\":\"scene\",\"chef_id\":-123,\"x\":" + x + ",\"y\":" + y + ",\"duration_ms\":" + duration + "}";
    }
    static Pending Next(Transport transport)
    {
        for (int i = 0; i < 200; i++)
        {
            Pending p = transport.Next();
            if (p != null) return p;
            Thread.Sleep(5);
        }
        throw new Exception("No request queued");
    }
    static void Send(TcpClient client, string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(value);
        client.GetStream().Write(bytes, 0, bytes.Length);
    }
    static string Read(TcpClient client)
    {
        client.ReceiveTimeout = 2000;
        var value = new StringBuilder();
        while (true)
        {
            int ch = client.GetStream().ReadByte();
            if (ch < 0 || ch == '\n') return value.ToString();
            value.Append((char)ch);
        }
    }
    static void Main()
    {
        string encoded = JsonOutput.Write(new StateReply());
        Check(encoded == "{\"ok\":true,\"state\":{\"name\":\"chef\\u000a\\\"1\\\"\",\"ids\":[1,-2]},\"before\":null}", "nested state JSON is preserved");
        System.Globalization.CultureInfo.CurrentCulture = new System.Globalization.CultureInfo("fr-FR");
        Check(JsonOutput.Write(new double[] { 1.25, -3.5 }) == "[1.25,-3.5]", "JSON numbers independent of system locale");
        Check(Command.Parse("{\"op\":\"state\"}").Op == "state", "state");
        Check(Command.Parse("{\"op\":\"release\"}").Op == "release", "release");
        Check(Command.Parse(Move("1")).DurationMs == 1, "minimum duration");
        Check(Command.Parse(Move("500")).DurationMs == 500, "maximum duration");
        Check(Command.Parse(Move("250", "0.6", "0.8")).ChefId == -123, "diagonal / negative Unity ID");
        Check(Command.Parse("{\"op\":\"pickup\",\"scene_token\":\"scene\",\"chef_id\":42}").Op == "pickup", "pickup");
        Check(Command.Parse("{\"op\":\"st\\u0061te\",\"request_id\":\"a\\\"b\"}").RequestId == "a\"b", "JSON escapes");
        foreach (string duration in new[] { "0", "501", "-1", "2.5", "1e3", "9999999999999999", "\"250\"", "null" }) Reject(Move(duration));
        foreach (string x in new[] { "NaN", "Infinity", "1e99", "1.1", "-1.1", "true", "01", ".5", "1." }) Reject(Move("250", x));
        Reject(Move("250", "1", "1")); Reject(Move("250", "0", "0"));
        foreach (string bad in new[] {
            "", "[]", "null", "{}", "{", "{\"op\":\"state\",}",
            "{\"op\":\"state\"}garbage", "{\"op\":\"state\"}\n{}",
            "{\"op\":\"state\",\"op\":\"pickup\"}", "{\"op\":\"unknown\"}",
            "{\"op\":\"move\"}", "{\"op\":\"pickup\",\"chef_id\":1}",
            "{\"op\":\"pickup\",\"scene_token\":\"s\",\"chef_id\":0}",
            "{\"op\":\"state\",\"typo\":1}", "{\"op\":\"st\nate\"}",
            "{\"op\":\"st\\qate\"}", "{\"op\":\"st\\uZZZZate\"}"
        }) Reject(bad);

        double now = 100;
        bool connected = true;
        var window = new ActionWindow(() => now, () => connected, 500);
        Check(window.Live, "input starts live"); now = 100.499;
        Check(window.Live, "input within deadline"); now = 100.5;
        Check(!window.Live, "input stops at deadline");
        now = 100;
        window = new ActionWindow(() => now, () => connected, 500);
        connected = false; Check(!window.Live, "disconnect immediately gates input");
        connected = true; window.Released = true; Check(!window.Live, "explicit release gates input");

        using (var server = new Transport(0))
        {
            var client = new TcpClient("127.0.0.1", server.Port);
            Send(client, "{\"op\":\"sta"); Thread.Sleep(20);
            Check(server.Next() == null, "partial line is not executed");
            Send(client, "te\"}\n");
            Pending pending = Next(server);
            Check(Command.Parse(pending.Line).Op == "state", "split TCP frame reassembled");
            pending.Finish("{\"ok\":true}");
            Check(Read(client) == "{\"ok\":true}", "newline-delimited response");
            Thread.Sleep(350); // Idle clients must survive a read timeout.
            Send(client, "invalid\n"); pending = Next(server);
            try { Command.Parse(pending.Line); throw new Exception("bad request parsed"); }
            catch (ArgumentException) { pending.Finish("{\"ok\":false,\"error\":\"invalid_request\"}"); }
            Check(Read(client).Contains("invalid_request"), "malformed message rejected without worker crash");
            Send(client, "{\"op\":\"state\"}\n"); pending = Next(server);
            var liveInput = new ActionWindow(() => now, () => pending.Peer.Connected, 500);
            client.Close();
            for (int i = 0; i < 100 && pending.Peer.Connected; i++) Thread.Sleep(5);
            Check(!pending.Peer.Connected && pending.Canceled, "disconnect cancels queued request");
            Check(!liveInput.Live, "real TCP disconnect gates active input");
            using (client = new TcpClient("127.0.0.1", server.Port))
            {
                Send(client, new string('x', 8193) + "\n");
                Check(Read(client).Contains("frame_too_large"), "oversized frame rejected");
            }
            using (client = new TcpClient("127.0.0.1", server.Port))
            {
                byte[] invalidUtf8 = { 0xff, 10 };
                client.GetStream().Write(invalidUtf8, 0, invalidUtf8.Length);
                Check(Read(client).Contains("invalid_utf8"), "invalid UTF-8 rejected");
            }
        }
        Console.WriteLine("PASS: " + checks + " protocol, deadline, and TCP checks (same core sources as plugin).");
    }
}
