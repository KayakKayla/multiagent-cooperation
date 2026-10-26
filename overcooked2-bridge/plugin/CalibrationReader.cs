using System;
using System.Collections.Generic;
using UnityEngine;

namespace Sims.Overcooked2Bridge
{
    // Read-only access to exact fields checked against this game's assemblies.
    // No state setters, interaction calls, or game methods that advance work.
    static class CalibrationReader
    {
        public static T Read<T>(object source, string name)
        { return FieldRead.Value<T>(source, name); }
        public static WashProgress Washing(GameObject go)
        {
            var settings = go.GetComponent<WashingStation>();
            var server = go.GetComponent<ServerWashingStation>();
            if (settings == null || server == null) return null;
            float elapsed = Read<float>(server, "m_cleaningTimer");
            var interactable = go.GetComponent<ServerInteractable>();
            return new WashProgress {
                elapsed = KitchenValues.Finite(elapsed), required = KitchenValues.Finite(settings.m_cleanPlateTime),
                completion_fraction = KitchenValues.Completion(elapsed, settings.m_cleanPlateTime),
                plate_count = Read<int>(server, "m_plateCount"),
                interactor_count = interactable == null ? 0 : interactable.InteractorCount()
            };
        }
        public static PlateReturnSnapshot PlateReturn()
        {
            var flows = UnityEngine.Object.FindObjectsOfType<ServerKitchenFlowControllerBase>();
            if (flows.Length != 1) return null;
            object controller = Read<object>(flows[0], "m_plateReturnController");
            if (controller == null) return null;
            object settings = Read<object>(controller, "m_plateReturnControllerDesc");
            object queue = Read<object>(controller, "m_platesToReturn");
            // This game-runtime FastList has Count and _items, but no Item indexer.
            Type type = queue.GetType();
            int count = (int)type.GetProperty("Count").GetValue(queue, null);
            if (count > 128) throw new InvalidOperationException("Plate return queue exceeds capture limit");
            var entries = new List<PendingPlateReturn>();
            for (int i = 0; i < count; i++)
            {
                object entry = FieldRead.ArrayEntry(queue, "_items", i);
                var station = Read<ServerPlateReturnStation>(entry, "m_station");
                entries.Add(new PendingPlateReturn {
                    station_id = station == null ? (int?)null : station.gameObject.GetInstanceID(),
                    remaining_seconds = KitchenValues.Finite(Read<float>(entry, "m_timer"))
                });
            }
            return new PlateReturnSnapshot { configured_seconds = KitchenValues.Finite(Read<float>(settings, "m_plateReturnTime")), pending = entries.ToArray() };
        }
    }
}
