"""Dataset configurations."""

DATASETS = {
    'Salinas': (
        'Salinas_corrected.mat',
        'Salinas_gt.mat',
        'salinas_corrected',
        'salinas_gt'
    ),
    'Indian_Pines': (
        'Indian_pines_corrected.mat',
        'Indian_pines_gt.mat',
        'indian_pines_corrected',
        'indian_pines_gt'
    ),
    'PaviaU': (
        'PaviaU.mat',
        'PaviaU_gt.mat',
        'paviaU',
        'paviaU_gt'
    ),
    'Pavia': (
        'Pavia.mat',
        'Pavia_gt.mat',
        'pavia',
        'pavia_gt'
    ),
    'KSC': (
        'KSC.mat',
        'KSC_gt.mat',
        'KSC',
        'KSC_gt'
    ),
    'Botswana': (
        'Botswana.mat',
        'Botswana_gt.mat',
        'Botswana',
        'Botswana_gt'
    )
}

SALINAS_CLASS_LABELS = {
    1: 'Broccoli_green_weeds_1',
    2: 'Broccoli_green_weeds_2',
    3: 'Fallow',
    4: 'Fallow_rough_plow',
    5: 'Fallow_smooth',
    6: 'Stubble',
    7: 'Celery',
    8: 'Grapes_untrained',
    9: 'Soil_vinyard_develop',
    10: 'Corn_senesced_green_weeds',
    11: 'Lettuce_romaine_4wk',
    12: 'Lettuce_romaine_5wk',
    13: 'Lettuce_romaine_6wk',
    14: 'Lettuce_romaine_7wk',
    15: 'Vinyard_untrained',
    16: 'Vinyard_vertical_trellis'
}

INDIAN_PINES_CLASS_LABELS = {
    1: 'Alfalfa',
    2: 'Corn-notill',
    3: 'Corn-mintill',
    4: 'Corn',
    5: 'Grass-pasture',
    6: 'Grass-trees',
    7: 'Grass-pasture-mowed',
    8: 'Hay-windrowed',
    9: 'Oats',
    10: 'Soybean-notill',
    11: 'Soybean-mintill',
    12: 'Soybean-clean',
    13: 'Wheat',
    14: 'Woods',
    15: 'Buildings-Grass-Trees-Drives',
    16: 'Stone-Steel-Towers'
}

PAVIAU_CLASS_LABELS = {
    1: 'Asphalt',
    2: 'Meadows',
    3: 'Gravel',
    4: 'Trees',
    5: 'Painted metal sheets',
    6: 'Bare Soil',
    7: 'Bitumen',
    8: 'Self-Blocking Bricks',
    9: 'Shadows'
}

PAVIA_CLASS_LABELS = {
    1: 'Water',
    2: 'Trees',
    3: 'Asphalt',
    4: 'Self-Blocking Bricks',
    5: 'Bitumen',
    6: 'Tiles',
    7: 'Shadows',
    8: 'Meadows',
    9: 'Bare Soil'
}

KSC_CLASS_LABELS = {
    1: 'Scrub',
    2: 'Willow swamp',
    3: 'CP hammock',
    4: 'Slash pine',
    5: 'Oak/Broadleaf',
    6: 'Hardwood',
    7: 'Swamp',
    8: 'Graminoid marsh',
    9: 'Spartina marsh',
    10: 'Cattail marsh',
    11: 'Salt marsh',
    12: 'Mud flats',
    13: 'Water'
}

BOTSWANA_CLASS_LABELS = {
    1: 'Water',
    2: 'Hippo grass',
    3: 'Flood Plain Grasses 1',
    4: 'Flood Plain Grasses 2',
    5: 'Reeds',
    6: 'Riparian',
    7: 'Fire scar',
    8: 'Island interior',
    9: 'Acacia woodlands',
    10: 'Acacia shrublands',
    11: 'Acacia grasslands',
    12: 'Short mopane',
    13: 'Mixed mopane',
    14: 'Exposed soils'
}

# Class label mappings
CLASS_LABELS_MAP = {
    'Salinas': SALINAS_CLASS_LABELS,
    'Indian_Pines': INDIAN_PINES_CLASS_LABELS,
    'PaviaU': PAVIAU_CLASS_LABELS,
    'Pavia': PAVIA_CLASS_LABELS,
    'KSC': KSC_CLASS_LABELS,
    'Botswana': BOTSWANA_CLASS_LABELS
}
