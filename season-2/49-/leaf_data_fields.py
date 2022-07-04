"""
Defined Washington State Leaf Data Systems dataset fields.
Cannabis Data Science Meetup Group
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 1/18/2022
Updated: 1/19/2022
License: MIT License <https://opensource.org/licenses/MIT>
"""

#------------------------------------------------------------------------------
# Lab result fields.
#------------------------------------------------------------------------------

lab_result_fields = {
    'global_id' : 'string',
    'mme_id' : 'string',
    'intermediate_type' : 'category',
    'status' : 'category',
    'global_for_inventory_id': 'string',
    'cannabinoid_status' : 'category',
    'cannabinoid_cbc_percent' : 'float16',
    'cannabinoid_cbc_mg_g' : 'float16',
    'cannabinoid_cbd_percent' : 'float16',
    'cannabinoid_cbd_mg_g' : 'float16',
    'cannabinoid_cbda_percent' : 'float16',
    'cannabinoid_cbda_mg_g' : 'float16',
    'cannabinoid_cbdv_percent' : 'float16',
    'cannabinoid_cbg_percent' : 'float16',
    'cannabinoid_cbg_mg_g' : 'float16',
    'cannabinoid_cbga_percent' : 'float16',
    'cannabinoid_cbga_mg_g' : 'float16',
    'cannabinoid_cbn_percent' : 'float16',
    'cannabinoid_cbn_mg_g' : 'float16',
    'cannabinoid_d8_thc_percent' : 'float16',
    'cannabinoid_d8_thc_mg_g' : 'float16',
    'cannabinoid_d9_thca_percent': 'float16',
    'cannabinoid_d9_thca_mg_g' : 'float16',
    'cannabinoid_d9_thc_percent' : 'float16',
    'cannabinoid_d9_thc_mg_g' : 'float16',
    'cannabinoid_thcv_percent' : 'float16',
    'cannabinoid_thcv_mg_g' : 'float16',
    'solvent_status' : 'category',
    'solvent_acetone_ppm' : 'float16',
    'solvent_benzene_ppm' : 'float16',
    'solvent_butanes_ppm' : 'float16',
    'solvent_chloroform_ppm' : 'float16',
    'solvent_cyclohexane_ppm' : 'float16',
    'solvent_dichloromethane_ppm' : 'float16',
    'solvent_ethyl_acetate_ppm' : 'float16',
    'solvent_heptane_ppm' : 'float16',
    'solvent_hexanes_ppm' : 'float16',
    'solvent_isopropanol_ppm' : 'float16',
    'solvent_methanol_ppm' : 'float16',
    'solvent_pentanes_ppm' : 'float16',
    'solvent_propane_ppm' : 'float16',
    'solvent_toluene_ppm' : 'float16',
    'solvent_xylene_ppm' : 'float16',
    'foreign_matter' : 'bool',
    'foreign_matter_stems': 'float16',
    'foreign_matter_seeds': 'float16',
    'microbial_status' : 'category',
    'microbial_bile_tolerant_cfu_g' : 'float16',
    'microbial_pathogenic_e_coli_cfu_g' : 'float16',
    'microbial_salmonella_cfu_g' : 'float16',
    'moisture_content_percent' : 'float16',
    'moisture_content_water_activity_rate' : 'float16',
    'mycotoxin_status' : 'category',
    'mycotoxin_aflatoxins_ppb' : 'float16',
    'mycotoxin_ochratoxin_ppb' : 'float16',
    'thc_percent' : 'float16',
    'notes' : 'float32',
    'testing_status' : 'category',
    'type' : 'category',
    'inventory_id' : 'string',
    'batch_id' : 'string',
    'parent_lab_result_id' : 'string',
    'og_parent_lab_result_id' : 'string',
    'copied_from_lab_id' : 'string',
    'external_id' : 'string',
    'lab_user_id' : 'string',
    'user_id' : 'string',
    'cannabinoid_editor' : 'float32',
    'microbial_editor' : 'string',
    'mycotoxin_editor' : 'string',
    'solvent_editor' : 'string',
}

lab_result_date_fields = [
    'created_at',
    'deleted_at',
    'updated_at',
    'received_at',
]

#------------------------------------------------------------------------------
# Inventories fields.
#------------------------------------------------------------------------------

licensee_fields = {
    'global_id' : 'string',
    'name': 'string',
    'type': 'string',
    'code': 'string',
    'address1': 'string',
    'address2': 'string',
    'city': 'string',
    'state_code': 'string',
    'postal_code': 'string',
    'country_code': 'string',
    'phone': 'string',
    'external_id': 'string',
    'certificate_number': 'string',
    'is_live': 'bool',
    'suspended': 'bool',
}

licensee_date_fields = [
    'created_at', # No records if issued before 2018-02-21.
    'updated_at',
    'deleted_at',
    'expired_at',
]

#------------------------------------------------------------------------------
# Inventories fields.
#------------------------------------------------------------------------------

inventory_fields = {
    'global_id' : 'string',
    'strain_id': 'string',
    'inventory_type_id': 'string',
    'qty': 'float16',
    'uom': 'string',
    'mme_id': 'string',
    'user_id': 'string',
    'external_id': 'string',
    'area_id': 'string',
    'batch_id': 'string',
    'lab_result_id': 'string',
    'lab_retest_id': 'string',
    'is_initial_inventory': 'bool',
    'created_by_mme_id': 'string',
    'additives': 'string',
    'serving_num': 'float16',
    'sent_for_testing': 'bool',
    'medically_compliant': 'string',
    'legacy_id': 'string',
    'lab_results_attested': 'int',
    'global_original_id': 'string',
}

inventory_date_fields = [
    'created_at', # No records if issued before 2018-02-21.
    'updated_at',
    'deleted_at',
    'inventory_created_at',
    'inventory_packaged_at',
    'lab_results_date',
]

#------------------------------------------------------------------------------
# Inventory type fields.
#------------------------------------------------------------------------------

inventory_type_fields = {
    'global_id': 'string',
    'mme_id': 'string',
    'user_id': 'string',
    'external_id': 'string',
    'uom': 'string',
    'name': 'string',
    'intermediate_type': 'string',
}

inventory_type_date_fields = [
    'created_at',
    'updated_at',
    'deleted_at',
]

#------------------------------------------------------------------------------
# Strain fields.
#------------------------------------------------------------------------------

strain_fields = {
    'mme_id': 'string',
    'user_id': 'string',
    'global_id': 'string',
    'external_id': 'string',
    'name': 'string',
}
strain_date_fields = [
    'created_at',
    'updated_at',
    'deleted_at',
]


#------------------------------------------------------------------------------
# Sales Items fields.
# TODO: Parse SalesItems_0, SalesItems_1, SalesItems_2, SalesItems_3
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Sales fields.
# TODO: Parse Sales_0, Sales_1, Sales_2
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Batches fields.
# TODO: Parse Batches_0
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Taxes fields.
# TODO: Parse Taxes_0
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Areas fields.
# TODO: Parse Areas_0
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
# Inventory Transfer Items fields.
# TODO: Parse InventoryTransferItems_0
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Inventory Transfer Items fields.
# TODO: Parse InventoryTransferItems_0
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Inventory Transfers fields.
# TODO: Parse InventoryTransfers_0
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Disposals fields.
# Optional: Parse Disposals_0
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Inventory Adjustments fields.
# Optional: Parse InventoryAdjustments_0, InventoryAdjustments_1, InventoryAdjustments_2
#------------------------------------------------------------------------------
