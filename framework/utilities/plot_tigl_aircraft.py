from framework.CPACS_update.cpacsfunctions import *
import numpy as np
from tigl3.configuration import CCPACSConfigurationManager_get_instance
from tigl3.boolean_ops import CFuseShapes
from tigl3.import_export_helper import export_shapes
import os

def get_fused_shape(tigl_handle):
    """
    This example uses the opencascade fusing algorithm to fuse wing
    and fuselage and writes the resulting geometry to a step file
    """
    print ("Computing fused geometry...")

    # get the configuration manager
    mgr = CCPACSConfigurationManager_get_instance()

    # get the CPACS configuration, defined by the tigl handle
    # we need to access the underlying tigl handle (that is used in the C/C++ API)
    config = mgr.get_configuration(tigl_handle._handle.value)

    fuselage_shape = config.get_fuselage(1).get_loft()
    wing_shape = config.get_wing(1).get_loft()
    wing_m_shape = config.get_wing(1).get_mirrored_loft()

    # fuse the shapes
    fused_shape = CFuseShapes(fuselage_shape, [wing_shape, wing_m_shape]).named_shape()

    print ("Done!")

    return fused_shape

def plot3d_tigl(vehicle):

    MODULE_DIR = 'c:/Users/aarc8/Documents/github\MDOAirB_base/framework/CPACS_update'
    cpacs_path = os.path.join(MODULE_DIR, 'ToolInput', 'Aircraft_In.xml')
    cpacs_out_path = os.path.join(MODULE_DIR, 'ToolOutput', 'Aircraft_Out.xml')
    tixi = open_tixi(cpacs_out_path)
    tigl = open_tigl(tixi)

    tixi_out = open_tixi(cpacs_out_path)
    wing = vehicle['wing']
    horizontal_tail = vehicle['horizontal_tail']
    vertical_tail = vehicle['vertical_tail']
    fuselage = vehicle['fuselage']
    engine = vehicle['engine']
    nacelle = vehicle['nacelle']
    aircraft = vehicle['aircraft']


    delta_x_root = 0
    delta_x_kink =  ((wing['semi_span_kink']*wing['semi_span'])-wing['root_chord_yposition'])*np.tan((wing['sweep_leading_edge']*np.pi)/180)
    delta_x_tip = (wing['semi_span']-(wing['semi_span_kink']*wing['semi_span']))*np.tan((wing['sweep_leading_edge']*np.pi)/180)

    # Wing ------------------------------------
    # center chord
    cc_w = wing['center_chord']
    xc_w = wing['leading_edge_xposition']
    yc_w = 0
    zc_w = 0
    # root chord
    cr_w = wing['root_chord']
    xr_w = wing['leading_edge_xposition'] + delta_x_root
    yr_w = wing['root_chord_yposition']
    zr_w = 0
    # kink chord
    ck_w = wing['kink_chord']
    xk_w = wing['leading_edge_xposition'] + delta_x_kink
    yk_w = wing['semi_span_kink']*wing['semi_span']
    zk_w = 0
    # tip chord
    ct_w = wing['tip_chord']
    xt_w = xr_w + delta_x_kink+ delta_x_tip
    yt_w = wing['semi_span']
    zt_w = 0 + wing['semi_span']*(np.tan((wing['dihedral']*np.pi)/180))

    # EH ------------------------------------
    # root chord
    cr_h = horizontal_tail['center_chord']
    xr_h = horizontal_tail['leading_edge_xposition']
    yr_h = 0
    zr_h = 0
    # tip chord
    ct_h = horizontal_tail['taper_ratio']*horizontal_tail['center_chord']
    yt_h = (horizontal_tail['span']/2)
    xt_h = xr_h + (cr_h - ct_h)/4 + yt_h*np.tan((horizontal_tail['sweep_c_4']*np.pi)/180)
    zt_h = 0

    # EV ------------------------------------
    # root chord
    cr_v = vertical_tail['center_chord']
    xr_v = vertical_tail['leading_edge_xposition']
    yr_v = 0
    zr_v = 0
    
    # tip chord
    ct_v = vertical_tail['tip_chord']
    xt_v = xr_v + cr_v/4 + vertical_tail['span']*np.tan((vertical_tail['sweep_c_4']*np.pi)/180) - ct_v/4
    yt_v = 0
    zt_v = vertical_tail['span']
    
    L_f = fuselage['length']
    D_f = fuselage['diameter']
    x_n = engine['center_of_gravity_xposition']
    y_n = engine['yposition']
    z_n = 0
    L_n = engine['length']
    D_n = engine['diameter'] 
    xcg_0 = aircraft['after_center_of_gravity_xposition']
    xnp = aircraft['neutral_point_xposition']


    len_hip = vertical_tail['span']/np.cos((vertical_tail['sweep_leading_edge']*np.pi)/180)



    fuselage_xpath = '/cpacs/vehicles/aircraft/model/fuselages/fuselage[1]/'
    
    # Update leading edge position
    tixi_out.updateDoubleElement(fuselage_xpath+'positionings/positioning[8]/length',fuselage['cabine_length']/4, '%g')
    tixi_out.updateDoubleElement(fuselage_xpath+'positionings/positioning[9]/length',fuselage['cabine_length']/4, '%g')
    tixi_out.updateDoubleElement(fuselage_xpath+'positionings/positioning[10]/length',fuselage['cabine_length']/4, '%g')
    tixi_out.updateDoubleElement(fuselage_xpath+'positionings/positioning[11]/length',fuselage['cabine_length']/4, '%g')

    tixi_out.updateDoubleElement(fuselage_xpath+'positionings/positioning[12]/length',fuselage['tail_length']/2, '%g')
    tixi_out.updateDoubleElement(fuselage_xpath+'positionings/positioning[13]/length',fuselage['tail_length']/2, '%g')

    nominal_diameter = 1.939*2
    scale_factor = fuselage['diameter']/nominal_diameter

    tixi_out.updateDoubleElement(fuselage_xpath+'transformation/scaling/y',scale_factor, '%g')
    tixi_out.updateDoubleElement(fuselage_xpath+'transformation/scaling/z',scale_factor, '%g')


    wing_xpath = '/cpacs/vehicles/aircraft/model/wings/wing[1]/'
    
    # tixi_out.updateDoubleElement(wing_xpath+'transformation/translation/x', wing['leading_edge_xposition'], '%g')
    tixi_out.updateDoubleElement(wing_xpath+'transformation/translation/z', -fuselage['diameter']/4, '%g')
    # Update center chord 
    tixi_out.updateDoubleElement(wing_xpath+'sections/section[1]/elements/element/transformation/scaling/x', cc_w, '%g')
    tixi_out.updateDoubleElement(wing_xpath+'sections/section[1]/elements/element/transformation/scaling/y', cc_w, '%g')
    tixi_out.updateDoubleElement(wing_xpath+'sections/section[1]/elements/element/transformation/scaling/z', cc_w, '%g')
    tixi_out.updateDoubleElement(wing_xpath+'sections/section[1]/transformation/translation/x', xc_w, '%g')
    tixi_out.updateDoubleElement(wing_xpath+'sections/section[1]/transformation/translation/y', yc_w, '%g')
    tixi_out.updateDoubleElement(wing_xpath+'sections/section[1]/transformation/translation/z', zc_w, '%g')


    # Update root chord 
    tixi_out.updateDoubleElement(wing_xpath+'sections/section[2]/elements/element/transformation/scaling/x', cr_w, '%g')
    tixi_out.updateDoubleElement(wing_xpath+'sections/section[2]/elements/element/transformation/scaling/y', cr_w, '%g')
    tixi_out.updateDoubleElement(wing_xpath+'sections/section[2]/elements/element/transformation/scaling/z', cr_w, '%g')
    tixi_out.updateDoubleElement(wing_xpath+'sections/section[2]/transformation/translation/x', xr_w, '%g')
    tixi_out.updateDoubleElement(wing_xpath+'sections/section[2]/transformation/translation/y', yr_w, '%g')
    tixi_out.updateDoubleElement(wing_xpath+'sections/section[2]/transformation/translation/z', zr_w, '%g')

    # Update kink chord 

    tixi_out.updateDoubleElement(wing_xpath+'sections/section[3]/elements/element/transformation/scaling/x', ck_w, '%g')
    tixi_out.updateDoubleElement(wing_xpath+'sections/section[3]/elements/element/transformation/scaling/y', ck_w, '%g')
    tixi_out.updateDoubleElement(wing_xpath+'sections/section[3]/elements/element/transformation/scaling/z', ck_w, '%g')

    tixi_out.updateDoubleElement(wing_xpath+'sections/section[3]/transformation/translation/x', xk_w, '%g')
    tixi_out.updateDoubleElement(wing_xpath+'sections/section[3]/transformation/translation/y', yk_w, '%g')
    tixi_out.updateDoubleElement(wing_xpath+'sections/section[3]/transformation/translation/z', zk_w, '%g')
    
    # Update tip chord 

    tixi_out.updateDoubleElement(wing_xpath+'sections/section[4]/elements/element/transformation/scaling/x', ct_w, '%g')
    tixi_out.updateDoubleElement(wing_xpath+'sections/section[4]/elements/element/transformation/scaling/y', ct_w, '%g')
    tixi_out.updateDoubleElement(wing_xpath+'sections/section[4]/elements/element/transformation/scaling/z', ct_w, '%g')

    tixi_out.updateDoubleElement(wing_xpath+'sections/section[4]/transformation/translation/x', xt_w, '%g')
    tixi_out.updateDoubleElement(wing_xpath+'sections/section[4]/transformation/translation/y', yt_w, '%g')
    tixi_out.updateDoubleElement(wing_xpath+'sections/section[4]/transformation/translation/z', zt_w, '%g')

    horizontal_thail_xpath = '/cpacs/vehicles/aircraft/model/wings/wing[2]/'

    # Update center chord 
    # tixi_out.updateDoubleElement(horizontal_thail_xpath+'transformation/translation/x', horizontal_tail['leading_edge_xposition'], '%g')
    tixi_out.updateDoubleElement(horizontal_thail_xpath+'sections/section[1]/elements/element/transformation/scaling/x', cr_h, '%g')
    tixi_out.updateDoubleElement(horizontal_thail_xpath+'sections/section[1]/elements/element/transformation/scaling/y', cr_h, '%g')
    tixi_out.updateDoubleElement(horizontal_thail_xpath+'sections/section[1]/elements/element/transformation/scaling/z', cr_h, '%g')
    tixi_out.updateDoubleElement(horizontal_thail_xpath+'sections/section[1]/transformation/translation/x', xr_h, '%g')
    tixi_out.updateDoubleElement(horizontal_thail_xpath+'sections/section[1]/transformation/translation/y', yr_h, '%g')
    tixi_out.updateDoubleElement(horizontal_thail_xpath+'sections/section[1]/transformation/translation/z', zr_h, '%g')


    tixi_out.updateDoubleElement(horizontal_thail_xpath+'sections/section[2]/elements/element/transformation/scaling/x', ct_h, '%g')
    tixi_out.updateDoubleElement(horizontal_thail_xpath+'sections/section[2]/elements/element/transformation/scaling/y', ct_h, '%g')
    tixi_out.updateDoubleElement(horizontal_thail_xpath+'sections/section[2]/elements/element/transformation/scaling/z', ct_h, '%g')
    tixi_out.updateDoubleElement(horizontal_thail_xpath+'sections/section[2]/transformation/translation/x', xt_h, '%g')
    tixi_out.updateDoubleElement(horizontal_thail_xpath+'sections/section[2]/transformation/translation/y', yt_h, '%g')
    tixi_out.updateDoubleElement(horizontal_thail_xpath+'sections/section[2]/transformation/translation/z', zt_h, '%g')

    # Update root chord 

    vertical_tail_xpath = '/cpacs/vehicles/aircraft/model/wings/wing[3]/'
    # vertical_tail['leading_edge_xposition'] = vertical_tail['aerodynamic_center_xposition'] - vertical_tail['center_chord']*0.25
    

    # # Update leading edge position
    tixi_out.updateDoubleElement(vertical_tail_xpath+'transformation/translation/x', vertical_tail['leading_edge_xposition'], '%g')
    # # Update center chord 
    tixi_out.updateDoubleElement(vertical_tail_xpath+'sections/section[1]/elements/element/transformation/scaling/x', cr_v, '%g')
    tixi_out.updateDoubleElement(vertical_tail_xpath+'sections/section[1]/elements/element/transformation/scaling/y', cr_v, '%g')
    tixi_out.updateDoubleElement(vertical_tail_xpath+'sections/section[1]/elements/element/transformation/scaling/z', cr_v, '%g')

    tixi_out.updateDoubleElement(vertical_tail_xpath+'sections/section[2]/elements/element/transformation/scaling/x', ct_v, '%g')
    tixi_out.updateDoubleElement(vertical_tail_xpath+'sections/section[2]/elements/element/transformation/scaling/y', ct_v, '%g')
    tixi_out.updateDoubleElement(vertical_tail_xpath+'sections/section[2]/elements/element/transformation/scaling/z', ct_v, '%g')
    # Update root chord 
    tixi_out.updateDoubleElement(vertical_tail_xpath+'positionings/positioning[2]/length',len_hip, '%g')
    tixi_out.updateDoubleElement(vertical_tail_xpath+'positionings/positioning[2]/sweepAngle',vertical_tail['sweep_leading_edge'], '%g')

    tixi_out = close_tixi(tixi_out, cpacs_out_path)

    tixi = open_tixi(cpacs_out_path)
    tigl = open_tigl(tixi)

        # Reference parameters
    Cref = tigl.wingGetMAC(tigl.wingGetUID(1))
    Sref = tigl.wingGetReferenceArea(1,1)
    b    = tigl.wingGetSpan(tigl.wingGetUID(1))

    wett    = tigl.wingGetWettedArea(tigl.wingGetUID(2))

    print('Cref',Cref)
    print('Sref',Sref)
    print('b',b)
    print('wetted',wett)

    # fused_ac = get_fused_shape(tigl)
    # export_shapes([fused_ac], "simpletest.stp")

    return


# import pickle

# # with open('Database/Family/40_to_100/all_dictionaries/'+str(36)+'.pkl', 'rb') as f:
# # with open('Database/Family/101_to_160/all_dictionaries/'+str(13)+'.pkl', 'rb') as f:
# # with open('Database/Family/161_to_220/all_dictionaries/'+str(8)+'.pkl', 'rb') as f:

# # with open('Database/Family_DD/40_to_100/all_dictionaries/'+str(28)+'.pkl', 'rb') as f:
# # with open('Database/Family_DD/101_to_160/all_dictionaries/'+str(2)+'.pkl', 'rb') as f:
# with open('Database/Family_DD/161_to_220/all_dictionaries/'+str(24)+'.pkl', 'rb') as f:


#     all_info_acft1 = pickle.load(f)
# #     all_info_acft1 = pickle.load(f)

# vehicle = all_info_acft1['vehicle']
# plot3d_tigl(vehicle)


# print(all_info_acft1['vehicle']['aircraft']['passenger_capacity'])
# print(all_info_acft1['vehicle']['fuselage']['seat_abreast_number'])
# print(all_info_acft1['vehicle']['wing']['area'])
# print(all_info_acft1['vehicle']['wing']['aspect_ratio'])
# print(all_info_acft1['vehicle']['wing']['semi_span_kink'])
# print(all_info_acft1['vehicle']['wing']['sweep_c_4'])
# print(all_info_acft1['vehicle']['wing']['taper_ratio'])
# print(all_info_acft1['vehicle']['wing']['twist'])
# print(all_info_acft1['vehicle']['engine']['type'])
# print(all_info_acft1['vehicle']['engine']['bypass'])
# print(all_info_acft1['vehicle']['engine']['fan_diameter'])
# print(all_info_acft1['vehicle']['engine']['maximum_thrust'])
# print(all_info_acft1['vehicle']['engine']['turbine_inlet_temperature'])
# print(all_info_acft1['vehicle']['performance']['range'])