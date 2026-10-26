using System;
using System.Collections.Generic;
using UnityEngine;

namespace Sims.Overcooked2Bridge
{
    // Read public game components/getters on Unity's main thread. This catalog is
    // deliberately partial: it is not a collision map or a simulator grid.
    static class KitchenReader
    {
        static Point3 Position(Vector3 v) { return new Point3(v.x, v.y, v.z); }

        public static KitchenSnapshot Capture(PlayerControls[] chefs)
        {
            var warnings = new List<string>();
            var stationObjects = new Dictionary<int, GameObject>();
            AddStations<StaticGridLocation>(stationObjects);
            AddStations<ServerAttachStation>(stationObjects);
            AddStations<ServerPickupItemSpawner>(stationObjects);
            AddStations<Workstation>(stationObjects);
            AddStations<CookingStation>(stationObjects);
            AddStations<HeatedStation>(stationObjects);
            AddStations<PlateStation>(stationObjects);
            AddStations<PlateReturnStation>(stationObjects);
            AddStations<PlateStackBase>(stationObjects);
            AddStations<WashingStation>(stationObjects);
            AddStations<IngredientDisposalBehaviour>(stationObjects);
            AddStations<ConveyorStation>(stationObjects);

            var itemObjects = new Dictionary<int, GameObject>();
            foreach (var attachment in UnityEngine.Object.FindObjectsOfType<ServerPhysicalAttachment>())
                AddItem(itemObjects, attachment.gameObject);
            var chefOwners = new Dictionary<int, int>();
            foreach (var chef in chefs)
            {
                var carrier = chef.GetComponent<ServerPlayerAttachmentCarrier>();
                if (carrier == null) continue;
                GameObject held = carrier.InspectCarriedItem();
                if (held != null) { AddItem(itemObjects, held); chefOwners[held.GetInstanceID()] = chef.GetInstanceID(); }
            }
            var stations = new List<StationSnapshot>();
            var stationOwners = new Dictionary<int, int>();
            foreach (GameObject go in Sorted(stationObjects))
            {
                var station = new StationSnapshot {
                    station_id = go.GetInstanceID(), name = go.name, scene = go.scene.name,
                    position = Position(go.transform.position), components = Components(go)
                };
                station.layer = go.layer;
                Observe(warnings, "work activity " + station.station_id, () => {
                    var interactable = go.GetComponent<ServerInteractable>();
                    if (interactable != null) station.work = new WorkActivity { interactor_count = interactable.InteractorCount() };
                });
                Observe(warnings, "washing " + station.station_id, () => {
                    station.washing = CalibrationReader.Washing(go);
                });
                var roles = new List<string>();
                if (go.GetComponent<Workstation>() != null) roles.Add("chopping_board");
                if (go.GetComponent<ServerPickupItemSpawner>() != null) roles.Add("dispenser");
                if (go.GetComponent<CookingStation>() != null) roles.Add("cooking_station");
                if (go.GetComponent<HeatedStation>() != null) roles.Add("heated_station");
                if (go.GetComponent<PlateStation>() != null) roles.Add("serving_station");
                if (go.GetComponent<PlateReturnStation>() != null) roles.Add("plate_return");
                if (go.GetComponent<PlateStackBase>() != null) roles.Add("plate_stack");
                if (go.GetComponent<WashingStation>() != null) roles.Add("washing_station");
                if (go.GetComponent<IngredientDisposalBehaviour>() != null) roles.Add("disposal");
                if (go.GetComponent<ConveyorStation>() != null) roles.Add("conveyor");
                if (go.GetComponent<ServerAttachStation>() != null) roles.Add("attachment_surface");
                if (roles.Count == 0) roles.Add("grid_location");
                station.roles = roles.ToArray();
                Observe(warnings, "station " + station.station_id, () => {
                    var grid = go.GetComponent<StaticGridLocation>();
                    if (grid != null && grid.AccessGridManager != null)
                        station.grid = new GridCell { grid_manager_id = grid.AccessGridManager.GetInstanceID(),
                            x = grid.GridIndex.X, y = grid.GridIndex.Y, z = grid.GridIndex.Z };
                    var surface = go.GetComponent<AttachStation>();
                    if (surface != null && surface.m_attachPoint != null)
                        station.attach_position = Position(surface.m_attachPoint.position);
                    var cooking = go.GetComponent<CookingStation>();
                    if (cooking != null) station.cooking_station_type = cooking.m_stationType.ToString();
                    var stack = go.GetComponent<ServerPlateStackBase>();
                    if (stack != null) station.plate_count = stack.GetSize();
                });
                Observe(warnings, "station contents " + station.station_id, () => {
                    var attach = go.GetComponent<ServerAttachStation>();
                    GameObject item = attach == null ? null : attach.InspectItem();
                    if (item != null)
                    {
                        int id = item.GetInstanceID();
                        station.item_id = id; AddItem(itemObjects, item);
                        if (stationOwners.ContainsKey(id) && stationOwners[id] != station.station_id)
                            warnings.Add("Multiple station owners for item " + id);
                        else stationOwners[id] = station.station_id;
                    }
                });
                Observe(warnings, "dispenser " + station.station_id, () => {
                    var dispenser = go.GetComponent<ServerPickupItemSpawner>();
                    GameObject prefab = dispenser == null ? null : dispenser.GetItemPrefab();
                    if (prefab != null)
                    {
                        station.dispenses = new ItemDefinition { prefab_name = prefab.name };
                        // Prefabs have not run server initialization: read only their
                        // static ingredient metadata, not live container handlers.
                        var ingredient = prefab.GetComponent<IngredientPropertiesComponent>();
                        if (ingredient != null)
                            station.dispenses.composition = Node(ingredient.GetOrderComposition(), 0, warnings);
                    }
                });
                stations.Add(station);
            }
            var items = new List<ItemSnapshot>();
            foreach (GameObject go in Sorted(itemObjects))
            {
                var item = new ItemSnapshot { item_id = go.GetInstanceID(), name = go.name,
                    scene = go.scene.name, position = Position(go.transform.position) };
                item.layer = go.layer;
                Observe(warnings, "plate stack " + item.item_id, () => {
                    var stack = go.GetComponent<ServerPlateStackBase>();
                    if (stack != null) item.plate_count = stack.GetSize();
                });
                int owner;
                if (chefOwners.TryGetValue(item.item_id, out owner)) item.held_by_chef_id = owner;
                if (stationOwners.TryGetValue(item.item_id, out owner)) item.station_id = owner;
                var roles = new List<string>();
                if (go.GetComponent<IngredientPropertiesComponent>() != null) roles.Add("ingredient");
                if (go.GetComponent<Plate>() != null) roles.Add("plate");
                if (go.GetComponent<IngredientContainer>() != null) roles.Add("ingredient_container");
                if (go.GetComponent<WorkableItem>() != null) roles.Add("workable");
                if (go.GetComponent<CookingHandler>() != null) roles.Add("cookable");
                item.roles = roles.ToArray();
                Observe(warnings, "item " + item.item_id, () => {
                    var physical = go.GetComponent<ServerPhysicalAttachment>();
                    if (physical != null) item.attached = physical.IsAttached();
                    item.composition = Composition(go, warnings);
                });
                item.location = KitchenValues.Location(item.attached, item.held_by_chef_id, item.station_id);
                if (item.location == "ambiguous") warnings.Add("Chef and station both reference item " + item.item_id);
                Observe(warnings, "ingredients " + item.item_id, () => {
                    var container = go.GetComponent<ServerIngredientContainer>();
                    if (container != null)
                    {
                        AssembledDefinitionNode[] contents = container.GetContents();
                        item.contents = Nodes(contents, 0, warnings);
                        item.contents_known = contents != null;
                    }
                });
                Observe(warnings, "chopping " + item.item_id, () => {
                    var server = go.GetComponent<ServerWorkableItem>();
                    var workable = go.GetComponent<WorkableItem>();
                    if (server != null && workable != null) item.chopping = new ChopProgress {
                        completion_fraction = KitchenValues.Finite(server.GetProgress()), stages = workable.m_stages,
                        chops_per_stage = CalibrationReader.Read<int>(server, "m_chopsPerSlice"),
                        next_prefab = workable.m_nextPrefab == null ? null : workable.m_nextPrefab.name
                    };
                });
                Observe(warnings, "cooking " + item.item_id, () => {
                    var server = go.GetComponent<ServerCookingHandler>();
                    if (server != null)
                    {
                        float elapsed = server.GetCookingProgress(), required = server.AccessCookingTime;
                        item.cooking = new CookProgress {
                            elapsed = KitchenValues.Finite(elapsed), required = KitchenValues.Finite(required),
                            completion_fraction = KitchenValues.Completion(elapsed, required),
                            state = server.GetCookedOrderState().ToString(),
                            required_station = server.GetRequiredStationType().ToString(),
                            is_cooked = server.IsCooked(), is_burning = server.IsBurning()
                        };
                    }
                });
                items.Add(item);
            }
            PlateReturnSnapshot plateReturn = null;
            Observe(warnings, "plate return timing", () => { plateReturn = CalibrationReader.PlateReturn(); });
            return new KitchenSnapshot { available = stations.Count > 0 || items.Count > 0, plate_return = plateReturn,
                stations = stations.ToArray(), items = items.ToArray(), warnings = warnings.ToArray() };
        }

        static void AddStations<T>(Dictionary<int, GameObject> objects) where T : Component
        {
            foreach (T component in UnityEngine.Object.FindObjectsOfType<T>())
            {
                GameObject go = component.gameObject;
                // Pots, plates, and other movable objects belong in items, not stations.
                if (go.GetComponent<PhysicalAttachment>() == null) objects[go.GetInstanceID()] = go;
            }
        }
        static void AddItem(Dictionary<int, GameObject> items, GameObject item)
        { if (item != null) items[item.GetInstanceID()] = item; }
        static List<GameObject> Sorted(Dictionary<int, GameObject> objects)
        {
            var result = new List<GameObject>(objects.Values);
            result.Sort((a, b) => a.GetInstanceID().CompareTo(b.GetInstanceID()));
            return result;
        }
        static string[] Components(GameObject go)
        {
            var names = new SortedDictionary<string, bool>();
            foreach (MonoBehaviour component in go.GetComponents<MonoBehaviour>())
                if (component != null && component.GetType().Assembly == typeof(PlayerControls).Assembly)
                    names[component.GetType().Name] = true;
            return new List<string>(names.Keys).ToArray();
        }
        static void Observe(List<string> warnings, string source, Action read)
        {
            try { read(); }
            catch (Exception ex) { warnings.Add(source + ": " + ex.GetType().Name + ": " + ex.Message); }
        }
        static RecipeNode Composition(GameObject go, List<string> warnings)
        {
            // Prefer a container's assembled contents over a base ingredient label.
            IOrderDefinition definition = go.GetComponent<ServerCookableContainer>();
            if (definition == null) definition = go.GetComponent<ServerPreparationContainer>();
            if (definition == null) definition = go.GetComponent<ServerPlate>();
            if (definition == null) definition = go.GetComponent<ServerCookableIngredient>();
            if (definition == null) definition = go.GetComponent<IngredientPropertiesComponent>();
            if (definition == null) definition = go.GetComponent<ServerItemContainer>();
            return definition == null ? null : Node(definition.GetOrderComposition(), 0, warnings);
        }
        static RecipeNode[] Nodes(AssembledDefinitionNode[] nodes, int depth, List<string> warnings)
        {
            if (nodes == null) return new RecipeNode[0];
            var result = new List<RecipeNode>();
            foreach (var node in nodes) { RecipeNode value = Node(node, depth, warnings); if (value != null) result.Add(value); }
            return result.ToArray();
        }
        static RecipeNode Node(AssembledDefinitionNode node, int depth, List<string> warnings)
        {
            if (node == null || node is NullAssembledNode) return null;
            var result = new RecipeNode { type = node.GetType().Name };
            if (depth >= 8)
            { result.truncated = true; warnings.Add("Composition depth limit reached"); return result; }
            var ingredient = node as IngredientAssembledNode;
            var item = node as ItemAssembledNode;
            OrderDefinitionNode definition = ingredient != null ? (OrderDefinitionNode)ingredient.m_ingriedientOrderNode : item != null ? item.m_itemOrderNode : null;
            if (definition != null) { result.definition_id = definition.m_uID; result.definition_name = definition.name; }
            var composite = node as CompositeAssembledNode;
            if (composite != null) result.contents = Nodes(composite.m_composition, depth + 1, warnings);
            var cooked = node as CookedCompositeAssembledNode;
            if (cooked != null)
            {
                result.cooking_state = cooked.m_progress.ToString();
                result.cooking_step = cooked.m_cookingStep == null ? null : cooked.m_cookingStep.name;
            }
            return result;
        }
    }
}
