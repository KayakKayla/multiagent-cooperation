using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;

namespace Sims.Overcooked2Bridge
{
    public sealed class Peer
    {
        public volatile bool Connected = true;
        public readonly TcpClient Client;
        public Peer(TcpClient client) { Client = client; }
        public void Close() { Connected = false; Client.Close(); }
    }
    public sealed class Pending
    {
        public string Line, Reply;
        public Peer Peer;
        public readonly ManualResetEvent Ready = new ManualResetEvent(false);
        public volatile bool Canceled;
        public void Finish(string reply) { Reply = reply; Ready.Set(); }
    }

    // The worker handles bytes only. Parsing, serialization, and Unity work happen
    // when the main thread dequeues a request; it never blocks on network reads.
    public sealed class Transport : IDisposable
    {
        readonly Queue<Pending> queue = new Queue<Pending>();
        readonly TcpListener listener;
        readonly Thread thread;
        volatile bool stopping;
        Peer current;
        public int Port { get { return ((IPEndPoint)listener.LocalEndpoint).Port; } }
        public Transport(int port)
        {
            listener = new TcpListener(IPAddress.Loopback, port);
            listener.Start(1);
            thread = new Thread(Run) { IsBackground = true, Name = "Sims OC2 bridge" };
            thread.Start();
        }
        public Pending Next()
        {
            lock (queue) return queue.Count == 0 ? null : queue.Dequeue();
        }
        void Run()
        {
            while (!stopping)
            {
                Peer peer = null;
                try
                {
                    peer = new Peer(listener.AcceptTcpClient());
                    current = peer;
                    peer.Client.NoDelay = true;
                    peer.Client.ReceiveTimeout = 250;
                    peer.Client.SendTimeout = 1000;
                    Serve(peer);
                }
                catch (SocketException) { }
                catch (IOException) { }
                catch (ObjectDisposedException) { }
                finally { if (peer != null) peer.Close(); current = null; }
            }
        }
        void Serve(Peer peer)
        {
            var stream = peer.Client.GetStream();
            var bytes = new List<byte>();
            while (!stopping && peer.Connected)
            {
                int value;
                try { value = stream.ReadByte(); }
                catch (IOException ex)
                {
                    var socket = ex.InnerException as SocketException;
                    if (socket != null && (socket.SocketErrorCode == SocketError.TimedOut || socket.SocketErrorCode == SocketError.WouldBlock)) continue;
                    throw;
                }
                if (value < 0) return;
                if (value != '\n')
                {
                    if (bytes.Count >= 8192) { Write(stream, "{\"ok\":false,\"error\":\"frame_too_large\"}"); return; }
                    bytes.Add((byte)value); continue;
                }
                string line;
                try { line = new UTF8Encoding(false, true).GetString(bytes.ToArray()); }
                catch (DecoderFallbackException) { Write(stream, "{\"ok\":false,\"error\":\"invalid_utf8\"}"); return; }
                bytes.Clear();
                var pending = new Pending { Line = line, Peer = peer };
                lock (queue) queue.Enqueue(pending);
                int waits = 0;
                while (!pending.Ready.WaitOne(10, false))
                {
                    // Detect EOF even while waiting for Unity's next update.
                    if (stopping || !peer.Connected || ++waits >= 200 ||
                        (peer.Client.Client.Poll(0, SelectMode.SelectRead) && peer.Client.Available == 0))
                    {
                        pending.Canceled = true;
                        peer.Connected = false;
                        return;
                    }
                }
                // The main thread has finished with the wait handle now.
                pending.Ready.Close();
                Write(stream, pending.Reply);
            }
        }
        static void Write(NetworkStream stream, string reply)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(reply + "\n");
            stream.Write(bytes, 0, bytes.Length);
        }
        public void Dispose()
        {
            stopping = true;
            listener.Stop();
            Peer peer = current;
            if (peer != null) peer.Close();
            thread.Join(1000);
            Pending request;
            while ((request = Next()) != null) { request.Canceled = true; request.Ready.Close(); }
        }
    }
}
