# Leaf

Introductory notes to the Leaf Data Systems API for Washington State cannabis traceability.

## Production

The first stage of the cannabis process is production.

<img src="./images/production-workflow.png" width="400px">


## Users

Before you can utilize the API, you will need to create users in the Leaf Data Systems UI.

<img src="./images/create_user.png" width="400px">

<img src="./images/users_table.png" width="400px">


## Areas

You will need to create areas to manage your inventory and production.


## Inventory Types

The are broad `type`s and specific `invetory_type`s.

### Types

- **Immature Plants**
    * `name` created automatically.
    * `uom` will be `ea`.
    * The *propagation source* determines the *sub-category*.

- **Mature Plants**
    * `name` created automatically.



"harvest_materials",
"intermediate_product",
"end_product",
"waste"

### Intermediate Types

# intermediate_product
"marijuana_mix",
"nonsolvent_based_concentrate",
"hydrocarbon_concentrate",
"co2_concentrate",
"ethanol_concentrate",
"food_grade_solvent_concentrate",
"infused_cooking_medium",

# end_product
"liquid_edible",
"solid_edible",
"concentrate_for_inhalation",
"topical",
"infused_mix",
"packaged_marijuana_mix",
"sample_jar",
"usable_marijuana",
"capsules",
"tinctures",
"transdermal_patches",
"suppositories",

# immature_plant
"seeds",
"clones",
"plant_tissue",

# mature_plant
"non_mandatory_plant_sample",

# harvest_materials
"flower",
"other_material",
"flower_lots",

# other_material_lots
"waste"  
