# Materials

A material decides how a part looks and, through its friction, how it touches. Imported robots and
environments bring their own materials; you can change them, make new ones and assign them per part.

## Where materials show

- **Properties → Materials** lists the drawn parts of the selected node, each with the material it uses.
  Pick another from a part's menu to reassign that part.
- **Properties → Material** edits one material: the selected part's, or the one picked at the top of the
  section.
- **Assets → Materials** lists every material of the scene with how many shapes use it, a search box and
  **+** to create one.

## Creating a material

**+** in the Assets panel's Materials page, or **+** beside a part's material in Properties ("New
material for this part"), offers:

- **Blank**, or a preset: **Plastic, Metal, Rubber, Glass, Emissive, Textured**;
- **Copy of** the material the part uses now.

A new material from Properties is assigned to that part at once. All of it is one undoable step.

## Editing a material

| Field | Meaning |
| --- | --- |
| **Base color** | The albedo, linear RGB and alpha. |
| **Roughness**, **Metallic** | PBR values from 0 to 1. |
| **Emissive** | Light the surface gives off. |
| **Maps** | Images for base color, roughness, metallic, normal and emissive. **Choose Image...** picks a file; a roughness or metallic map also says which image channel it reads. |
| **UV** / **Triplanar** | UV follows the mesh's texture coordinates; triplanar projects along the world axes, for meshes without them. |
| **Scale** | Repeats of the maps per UV unit, or per metre with triplanar. |
| **Offline beauty** | Transmission, IOR and sheen, which the path tracer uses. |
| **Friction** | The contact friction of every shape that uses the material. |

Drag an image file from the Assets panel onto a map slot to use it.

A material is **shared**: an edit changes every shape that uses it, and the section says how many do.
**Make Unique** gives the selected part a copy of its own first, so the edit changes that part only.

A collider takes its friction from its material until you set the friction on the collider itself
(**Properties → Collision**); from then on the collider keeps its own.

## Assigning

- Drag a material from the Assets panel onto a part in the viewport, or onto a row in the Stage panel.
  On a row it goes to every part under that node; on one of several selected nodes, to all of them.
- Right-click a material in the Assets panel for **Assign to Selection**, **Select Users**,
  **Duplicate**, **Rename**, **Copy Name** and **Delete**. Deleting a material in use asks what its users
  take instead.

## From scripts and agents

```python
app.actions.run("material.create", preset="metal",
                fields={"base_color": [0.8, 0.6, 0.2, 1.0]}, assign="/go2/base")
app.actions.run("material.set", material="Floor", roughness=0.7,
                albedo_map="C:/textures/concrete.png", uv_scale=4)
app.actions.run("material.assign", material="Rubber", paths=["/go2/FL_foot", "/go2/FR_foot"])
```

The [material actions](../reference/actions/materials.md) list every field.
