using System;

namespace Sims.Overcooked2Bridge
{
    // Plain DTOs: no Unity references escape into JSON or the socket worker.
    [Serializable] public class KitchenSnapshot
    {
        public int schema_version = 1;
        public bool available;
        public string coverage = "supported_components";
        public StationSnapshot[] stations = new StationSnapshot[0];
        public ItemSnapshot[] items = new ItemSnapshot[0];
        public string[] warnings = new string[0];
    }
    [Serializable] public class Point3
    {
        public float x, y, z;
        public Point3(float x, float y, float z) { this.x = x; this.y = y; this.z = z; }
    }
    [Serializable] public class GridCell
    {
        public int grid_manager_id, x, y, z;
    }
    [Serializable] public class StationSnapshot
    {
        public int station_id;
        public string name, scene;
        public Point3 position, attach_position;
        public GridCell grid;
        public string[] roles = new string[0], components = new string[0];
        public int? item_id, plate_count;
        public string cooking_station_type;
        public ItemDefinition dispenses;
    }
    [Serializable] public class ItemDefinition
    {
        public string prefab_name;
        public RecipeNode composition;
    }
    [Serializable] public class ItemSnapshot
    {
        public int item_id;
        public string name, scene;
        public Point3 position;
        public string[] roles = new string[0];
        public bool? attached;
        public string location;
        public int? held_by_chef_id, station_id;
        public bool contents_known;
        public RecipeNode[] contents = new RecipeNode[0];
        public RecipeNode composition;
        public ChopProgress chopping;
        public CookProgress cooking;
    }
    [Serializable] public class RecipeNode
    {
        public string type, definition_name, cooking_state, cooking_step;
        public int? definition_id;
        public bool truncated;
        public RecipeNode[] contents = new RecipeNode[0];
    }
    [Serializable] public class ChopProgress
    {
        public float? completion_fraction;
        public int stages;
        public string next_prefab;
    }
    [Serializable] public class CookProgress
    {
        public float? elapsed, required, completion_fraction;
        public string units = "game_cooking_time", state, required_station;
        public bool is_cooked, is_burning;
    }
    public static class KitchenValues
    {
        public static float? Finite(float value)
        { return float.IsNaN(value) || float.IsInfinity(value) ? (float?)null : value; }
        public static float? Completion(float elapsed, float required)
        {
            if (!Finite(elapsed).HasValue || !Finite(required).HasValue || required <= 0) return null;
            return Math.Max(0, Math.Min(1, elapsed / required));
        }
        public static string Location(bool? attached, int? chef, int? station)
        {
            if (chef.HasValue && station.HasValue) return "ambiguous";
            if (chef.HasValue) return "held";
            if (station.HasValue) return "station";
            return attached == false ? "unattached" : "attached_unknown";
        }
    }
}
