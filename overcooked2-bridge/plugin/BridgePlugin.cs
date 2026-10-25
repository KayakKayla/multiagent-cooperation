using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Security.Cryptography;
using BepInEx;
using UnityEngine;
using UnityEngine.SceneManagement;
using Team17.Online;

namespace Sims.Overcooked2Bridge
{
    [BepInPlugin("ca.waterloo.sims.overcooked2bridge", "Sims Overcooked2 Bridge", "0.1.0")]
    public class BridgePlugin : BaseUnityPlugin
    {
        public const string Version = "0.1.0";
        Transport transport;
        string sceneToken = Guid.NewGuid().ToString("N");
        string assemblyHash;
        ActiveInput active;
        bool previousRunInBackground;

        private void Awake()
        {
            Logger.LogInfo("SIMS_BRIDGE_STARTUP version=" + Version + " checkpoint=bridge");
            try
            {
                using (var hash = SHA256.Create())
                    assemblyHash = BitConverter.ToString(hash.ComputeHash(File.ReadAllBytes(typeof(PlayerControls).Assembly.Location))).Replace("-", "").ToLowerInvariant();
                int port = Config.Bind<int>("Bridge", "Port", 8765, "Loopback TCP port (restart game after changing).").Value;
                if (port < 1024 || port > 65535) throw new ArgumentException("Port must be 1024..65535");
                transport = new Transport(port);
                previousRunInBackground = Application.runInBackground;
                Application.runInBackground = true;
                SceneManager.sceneLoaded += SceneLoaded;
                SceneManager.sceneUnloaded += SceneUnloaded;
                Logger.LogInfo("SIMS_BRIDGE_READY address=127.0.0.1:" + port + " assembly_sha256=" + assemblyHash);
            }
            catch (Exception ex)
            {
                Logger.LogError("SIMS_BRIDGE_FAILED " + ex);
                enabled = false;
            }
        }

        static double Now() { return (double)Stopwatch.GetTimestamp() / Stopwatch.Frequency; }
        void SceneLoaded(Scene scene, LoadSceneMode mode) { SceneChanged(); }
        void SceneUnloaded(Scene scene) { SceneChanged(); }
        void SceneChanged()
        {
            EndInput("scene_changed");
            sceneToken = Guid.NewGuid().ToString("N");
        }

        void Update()
        {
            try
            {
                if (active != null && !InputAllowed(active)) EndInput("completed_or_canceled");
                Pending pending;
                // One client, one outstanding request; cap work per Unity frame.
                for (int i = 0; i < 4 && (pending = transport.Next()) != null; i++)
                {
                    if (pending.Canceled || !pending.Peer.Connected) { pending.Ready.Close(); continue; }
                    Reply reply = Handle(pending);
                    pending.Finish(JsonOutput.Write(reply));
                }
            }
            catch (Exception ex)
            {
                EndInput("exception");
                Logger.LogError("SIMS_BRIDGE_ERROR " + ex);
            }
        }

        Reply Handle(Pending pending)
        {
            var reply = new Reply { plugin_version = Version, assembly_sha256 = assemblyHash };
            try
            {
                Command command = Command.Parse(pending.Line);
                reply.request_id = command.RequestId;
                if (command.Op == "release") EndInput("release");
                else if (command.Op != "state")
                {
                    if (command.SceneToken != sceneToken) return Fail(reply, "stale_scene", "Read state again after a scene change.");
                    if (active != null) return Fail(reply, "busy", "Release or wait for the current input to finish.");
                    PlayerControls selected = null;
                    foreach (var chef in FindChefs()) if (chef.GetInstanceID() == command.ChefId) selected = chef;
                    if (selected == null) return Fail(reply, "chef_unavailable", "The listed chef is no longer available.");
                    if (!CanControl(selected)) return Fail(reply, "chef_not_ready", "Use one local player, select this chef, and resume the level.");
                    if (command.Op == "pickup")
                    {
                        var carrier = selected.GetComponent<ServerPlayerAttachmentCarrier>();
                        if (carrier == null) return Fail(reply, "carrier_unavailable", "Held-item state is unavailable.");
                        if (carrier.InspectCarriedItem() != null) return Fail(reply, "hands_not_empty", "Use an empty-handed chef for this pickup test.");
                    }
                    reply.before = Capture();
                    active = new ActiveInput(selected, command, pending.Peer, Now, CanControl);
                    Logger.LogInfo("SIMS_BRIDGE_INPUT op=" + command.Op + " chef=" + command.ChefId + " duration_ms=" + active.DurationMs);
                }
                reply.ok = true;
                reply.state = Capture();
            }
            catch (ArgumentException ex) { return Fail(reply, "invalid_request", ex.Message); }
            catch (Exception ex)
            {
                EndInput("exception");
                Logger.LogError(ex);
                return Fail(reply, "game_error", ex.Message);
            }
            return reply;
        }

        Reply Fail(Reply reply, string error, string message)
        {
            reply.error = error; reply.message = message;
            return reply;
        }

        static PlayerControls[] FindChefs()
        {
            var result = UnityEngine.Object.FindObjectsOfType<PlayerControls>();
            Array.Sort(result, (a, b) => a.GetInstanceID().CompareTo(b.GetInstanceID()));
            return result;
        }
        static bool LocalSession()
        {
            return ServerUserSystem.m_Users != null && ServerUserSystem.m_Users.Count == 1 && ServerUserSystem.m_Users._items[0].IsLocal;
        }
        static bool Paused()
        {
            return Time.timeScale == 0 || TimeManager.IsPaused(TimeManager.PauseLayer.Main) ||
                (T17InGameFlow.Instance != null && T17InGameFlow.Instance.IsPauseMenuOpen());
        }
        static bool InRound()
        {
            IFlowController flow = GameUtils.GetFlowController();
            return flow != null && flow.InRound;
        }
        static bool CanControl(PlayerControls chef)
        {
            return chef != null && chef.isActiveAndEnabled && chef.ControlScheme != null &&
                chef.PlayerIDProvider != null && chef.PlayerIDProvider.IsLocallyControlled() &&
                chef.GetDirectlyUnderPlayerControl() && chef.CanButtonBePressed() &&
                LocalSession() && InRound() && !Paused();
        }
        bool InputAllowed(ActiveInput input)
        {
            return input.Window.Live && CanControl(input.Chef) && input.Chef.ControlScheme == input.Scheme;
        }
        Snapshot Capture()
        {
            var chefs = new List<ChefState>();
            foreach (PlayerControls chef in FindChefs())
            {
                var held = new HeldState();
                var carrier = chef.GetComponent<ServerPlayerAttachmentCarrier>();
                if (carrier != null)
                {
                    held.known = true;
                    GameObject item = carrier.InspectCarriedItem();
                    held.occupied = item != null;
                    if (item != null) { held.instance_id = item.GetInstanceID(); held.name = item.name; }
                }
                chefs.Add(new ChefState {
                    chef_id = chef.GetInstanceID(), name = chef.name,
                    player_id = chef.PlayerIDProvider == null ? "" : chef.PlayerIDProvider.GetID().ToString(),
                    position = new Vec(chef.transform.position), facing = new Vec(chef.transform.forward),
                    selected = chef.GetDirectlyUnderPlayerControl(), can_control = CanControl(chef), held = held
                });
            }
            return new Snapshot {
                scene = SceneManager.GetActiveScene().name, scene_token = sceneToken,
                frame = Time.frameCount, monotonic_seconds = Now(),
                paused = Paused(), in_round = InRound(), local_single_player = LocalSession(),
                input_active = active != null && active.Window.Live,
                active_chef_id = active == null ? 0 : active.ChefId, chefs = chefs.ToArray()
            };
        }
        void EndInput(string reason)
        {
            if (active == null) return;
            active.Restore();
            active = null;
            Logger.LogInfo("SIMS_BRIDGE_RELEASE reason=" + reason);
        }
        void OnDisable() { EndInput("disabled"); }
        void OnDestroy()
        {
            EndInput("shutdown");
            SceneManager.sceneLoaded -= SceneLoaded;
            SceneManager.sceneUnloaded -= SceneUnloaded;
            if (transport != null) { transport.Dispose(); Application.runInBackground = previousRunInBackground; }
        }
    }

    // Save and restore exactly the inputs we replace. Game timing and mechanics
    // continue normally; wrappers are sampled by the game's own control loop.
    sealed class ActiveInput
    {
        public readonly PlayerControls Chef;
        public readonly PlayerControls.ControlSchemeData Scheme;
        public readonly int ChefId, DurationMs;
        public readonly ActionWindow Window;
        readonly ILogicalValue oldX, oldY;
        readonly ILogicalButton oldPickup;
        readonly Axis x, y;
        readonly Pulse pickup;
        public ActiveInput(PlayerControls chef, Command command, Peer peer, Func<double> clock, Func<PlayerControls, bool> ready)
        {
            Chef = chef; ChefId = chef.GetInstanceID(); Scheme = chef.ControlScheme;
            DurationMs = command.Op == "move" ? command.DurationMs : 150;
            Window = new ActionWindow(clock, () => peer.Connected, DurationMs);
            oldX = Scheme.m_moveX; oldY = Scheme.m_moveY; oldPickup = Scheme.m_pickupButton;
            Func<bool> live = () => Window.Live && ready(chef) && chef.ControlScheme == Scheme;
            x = new Axis(oldX, live, command.Op == "move" ? command.X : 0);
            y = new Axis(oldY, live, command.Op == "move" ? command.Y : 0);
            pickup = new Pulse(oldPickup, live, command.Op == "pickup");
            Scheme.m_moveX = x; Scheme.m_moveY = y; Scheme.m_pickupButton = pickup;
        }
        public void Restore()
        {
            Window.Released = true;
            if (ReferenceEquals(Scheme.m_moveX, x)) Scheme.m_moveX = oldX;
            if (ReferenceEquals(Scheme.m_moveY, y)) Scheme.m_moveY = oldY;
            if (ReferenceEquals(Scheme.m_pickupButton, pickup)) Scheme.m_pickupButton = oldPickup;
        }
    }
    sealed class Axis : ILogicalValue
    {
        readonly ILogicalValue original;
        readonly Func<bool> live;
        readonly float value;
        public Axis(ILogicalValue original, Func<bool> live, float value) { this.original = original; this.live = live; this.value = value; }
        public float GetValue() { return live() ? value : 0; }
        public void GetLogicTreeData(out AcyclicGraph<ILogicalElement, LogicalLinkInfo> graph, out AcyclicGraph<ILogicalElement, LogicalLinkInfo>.Node head)
        { original.GetLogicTreeData(out graph, out head); }
    }
    sealed class Pulse : ILogicalButton
    {
        readonly ILogicalButton original;
        readonly Func<bool> live;
        readonly bool enabled;
        int frame = -1;
        bool claimedPress, claimedRelease;
        public Pulse(ILogicalButton original, Func<bool> live, bool enabled) { this.original = original; this.live = live; this.enabled = enabled; }
        int FirstFrame() { if (frame < 0 && enabled && live()) frame = Time.frameCount; return frame; }
        public bool JustPressed() { int first = FirstFrame(); return enabled && live() && Time.frameCount == first; }
        public bool JustReleased() { return frame >= 0 && live() && Time.frameCount == frame + 1; }
        public bool IsDown() { return JustPressed(); }
        public float GetHeldTimeLength() { return 0; }
        public bool HasUnclaimedPressEvent() { return !claimedPress && JustPressed(); }
        public bool HasUnclaimedReleaseEvent() { return !claimedRelease && JustReleased(); }
        public void ClaimPressEvent() { claimedPress = true; }
        public void ClaimReleaseEvent() { claimedRelease = true; }
        public void GetLogicTreeData(out AcyclicGraph<ILogicalElement, LogicalLinkInfo> graph, out AcyclicGraph<ILogicalElement, LogicalLinkInfo>.Node head)
        { original.GetLogicTreeData(out graph, out head); }
    }

    [Serializable] public class Reply
    {
        public bool ok;
        public string request_id = "", error = "", message = "", plugin_version, assembly_sha256;
        public Snapshot before, state;
    }
    [Serializable] public class Snapshot
    {
        public string scene, scene_token;
        public int frame, active_chef_id;
        public double monotonic_seconds;
        public bool paused, in_round, local_single_player, input_active;
        public ChefState[] chefs;
    }
    [Serializable] public class ChefState
    {
        public int chef_id;
        public string name, player_id;
        public Vec position, facing;
        public bool selected, can_control;
        public HeldState held;
    }
    [Serializable] public class HeldState
    {
        public bool known, occupied;
        public int instance_id;
        public string name = "";
    }
    [Serializable] public class Vec
    {
        public float x, y, z;
        public Vec(Vector3 v) { x = v.x; y = v.y; z = v.z; }
    }
}
