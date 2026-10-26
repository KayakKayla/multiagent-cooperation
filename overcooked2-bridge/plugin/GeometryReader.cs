using System;
using System.Collections.Generic;
using UnityEngine;

namespace Sims.Overcooked2Bridge
{
    [Serializable] public class GeometrySnapshot
    {
        public string coverage = "active_collider_world_bounds";
        public string limitation = "Bounds are not exact collision shapes or a verified walkability grid.";
        public ColliderSnapshot[] colliders;
        public string[] warnings;
    }
    [Serializable] public class ColliderSnapshot
    {
        public int collider_id, object_id, layer;
        public string name, scene, type, layer_name;
        public bool trigger, dynamic_body;
        public Point3 center, size;
    }
    static class GeometryReader
    {
        // Captured once per scene by the recorder, not during each 10 Hz poll.
        public static GeometrySnapshot Capture()
        {
            var result = new List<ColliderSnapshot>();
            var warnings = new List<string>();
            foreach (var collider in UnityEngine.Object.FindObjectsOfType<Collider>())
            {
                if (!collider.enabled || !collider.gameObject.activeInHierarchy) continue;
                if (result.Count >= 4096) { warnings.Add("Collider capture truncated at 4096 entries"); break; }
                var go = collider.gameObject;
                Bounds b = collider.bounds;
                result.Add(new ColliderSnapshot {
                    collider_id = collider.GetInstanceID(), object_id = go.GetInstanceID(),
                    name = go.name, scene = go.scene.name, type = collider.GetType().Name,
                    layer = go.layer, layer_name = LayerMask.LayerToName(go.layer),
                    trigger = collider.isTrigger, dynamic_body = collider.attachedRigidbody != null && !collider.attachedRigidbody.isKinematic,
                    center = new Point3(b.center.x, b.center.y, b.center.z), size = new Point3(b.size.x, b.size.y, b.size.z)
                });
            }
            result.Sort((a, b) => a.collider_id.CompareTo(b.collider_id));
            return new GeometrySnapshot { colliders = result.ToArray(), warnings = warnings.ToArray() };
        }
    }
}
