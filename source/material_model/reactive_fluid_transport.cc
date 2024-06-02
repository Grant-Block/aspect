/*
  Copyright (C) 2023 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.

  ASPECT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ASPECT; see the file LICENSE.  If not see
  <http://www.gnu.org/licenses/>.
*/

#include <aspect/material_model/reactive_fluid_transport.h>
#include <aspect/simulator_access.h>
#include <aspect/adiabatic_conditions/interface.h>
#include <aspect/gravity_model/interface.h>
#include <aspect/utilities.h>
#include <aspect/geometry_model/interface.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/numerics/fe_field_function.h>

namespace aspect
{
  namespace MaterialModel
  {
    template <int dim>
    bool
    ReactiveFluidTransport<dim>::
    is_compressible () const
    {
      return base_model->is_compressible();
    }



    template <int dim>
    double
    ReactiveFluidTransport<dim>::
    reference_darcy_coefficient () const
    {
      // 0.01 = 1% melt
      return reference_permeability * std::pow(0.01,3.0) / eta_f;
    }



    template <int dim>
    std::vector<double>
    ReactiveFluidTransport<dim>::
    tian_equilibrium_bound_water_content (const MaterialModel::MaterialModelInputs<dim> &in,
                                          unsigned int q) const
    {
      // Create arrays that will store the values of the polynomials at the current pressure
      std::vector<double> LR_values(4);
      std::vector<double> csat_values(4);
      std::vector<double> Td_values(4);

      // Loop over the four rock types (peridotite, gabbro, MORB, sediment) and the polynomial
      // coefficients to fill the vectors defined above. The polynomials for LR are defined in
      // equations 13, B2, B10, and B18. csat polynomials are defined in equations 14, B1, B9, and B17.
      // Td polynomials are defined in equations 15, B3, B11, and B19.
      for (unsigned int i = 0; i<devolatilization_enthalpy_changes.size(); ++i)
        {
          // Pressure, which must be in GPa for the parametrization, or GPa^-1. The polynomials for each lithology
          // breaks down above certain pressures, make sure that we cap the pressure just before this break down.
          // Introduce minimum pressure to avoid a division by 0.
          const double minimum_pressure = 1e-12;
          const double pressure = std::min(std::max(minimum_pressure, in.pressure[q]/1.e9), pressure_cutoffs[i]);
          const double inverse_pressure = 1.0/pressure;
          for (unsigned int j = 0; j<devolatilization_enthalpy_changes[i].size(); ++j)
            {
              LR_values[i] += devolatilization_enthalpy_changes[i][j] * std::pow(inverse_pressure, devolatilization_enthalpy_changes[i].size() - 1 - j);
            }

          for (unsigned int j = 0; j<water_mass_fractions[i].size(); ++j)
            {
              csat_values[i] += i==3 ? water_mass_fractions[i][j] * std::pow(std::log10(pressure), water_mass_fractions[i].size() - 1 - j) :\
                                water_mass_fractions[i][j] * std::pow(pressure, water_mass_fractions[i].size() - 1 - j);
            }

          for (unsigned int j = 0; j<devolatilization_onset_temperatures[i].size(); ++j)
            {
              Td_values[i] += devolatilization_onset_temperatures[i][j] * std::pow(pressure, devolatilization_onset_temperatures[i].size() - 1 - j);
            }
        }

      // Create an array for the equilibrium bound water content that is calculated from these polynomials
      std::vector<double> eq_bound_water_content(4);

      // Define the maximum bound water content allowed for the four different rock compositions
      std::vector<double> max_bound_water_content = {tian_max_peridotite_water, tian_max_gabbro_water, tian_max_MORB_water, tian_max_sediment_water};

      // Loop over all rock compositions and fill the equilibrium bound water content, divide by 100 to convert
      // from percentage to fraction (equation 1)
      for (unsigned int k = 0; k<LR_values.size(); ++k)
        {
          eq_bound_water_content[k] = (std::min(std::exp(csat_values[k]) * \
                                                std::exp(std::exp(LR_values[k]) * (1/in.temperature[q] - 1/Td_values[k])), \
                                                max_bound_water_content[k]) / 100.0);
        }
      return eq_bound_water_content;
    }



    template <int dim>
    void
    ReactiveFluidTransport<dim>::
    melt_fractions (const MaterialModel::MaterialModelInputs<dim> &in,
                    std::vector<double> &melt_fractions) const
    {
      for (unsigned int q=0; q<in.temperature.size(); ++q)
        {
          const unsigned int porosity_idx = this->introspection().compositional_index_for_name("porosity");
          switch (fluid_solid_reaction_scheme)
            {
              case no_reaction:
              {
                // No reactions occur between the solid and fluid phases,
                // and the fluid volume fraction (stored in the melt_fractions
                // vector) is equal to the porosity.
                melt_fractions[q] = in.composition[q][porosity_idx];
                break;
              }
              case zero_solubility:
              {
                // The fluid volume fraction in equilibrium with the solid
                // at any point (stored in the melt_fractions vector) is
                // equal to the sum of the bound fluid content and porosity.
                const unsigned int bound_fluid_idx = this->introspection().compositional_index_for_name("bound_fluid");
                melt_fractions[q] = in.composition[q][bound_fluid_idx] + in.composition[q][porosity_idx];
                break;
              }
              case tian_approximation:
              {
                // The bound fluid content is calculated using parametrized phase
                // diagrams for four different rock types: sediment, MORB, gabbro, and
                // peridotite.
                const unsigned int bound_fluid_idx = this->introspection().compositional_index_for_name("bound_fluid");
                const unsigned int sediment_idx = this->introspection().compositional_index_for_name("sediment");
                const unsigned int MORB_idx = this->introspection().compositional_index_for_name("MORB");
                const unsigned int gabbro_idx = this->introspection().compositional_index_for_name("gabbro");
                const unsigned int peridotite_idx = this->introspection().compositional_index_for_name("peridotite");

                // Initialize a vector that stores the compositions (mass fractions) for
                // the four different rock compositions,
                std::vector<double> tracked_rock_mass_fractions(4);
                tracked_rock_mass_fractions[0] = (in.composition[q][peridotite_idx]);
                tracked_rock_mass_fractions[1] = (in.composition[q][gabbro_idx]);
                tracked_rock_mass_fractions[2] = (in.composition[q][MORB_idx]);
                tracked_rock_mass_fractions[3] = (in.composition[q][sediment_idx]);

                // The bound water content (water within the solid phase) for the four different rock types
                std::vector<double> tian_eq_bound_water_content = tian_equilibrium_bound_water_content(in, q);

                // average the water content between the four different rock types
                double average_eq_bound_water_content = MaterialUtilities::average_value (tracked_rock_mass_fractions, tian_eq_bound_water_content, MaterialUtilities::arithmetic);

                // The fluid volume fraction in equilibrium with the solid (stored in the melt_fractions vector)
                // is equal to the sum of the porosity and the change in bound fluid content
                // (current bound fluid - updated average bound fluid).
                melt_fractions[q] = std::max(in.composition[q][bound_fluid_idx] + in.composition[q][porosity_idx] - average_eq_bound_water_content, 0.0);
                break;
              }
              case katz2003:
              {
                melt_fractions[q] = katz2003_model.melt_fraction(in.temperature[q],
                                                                 this->get_adiabatic_conditions().pressure(in.position[q]));
                break;
              }
              default:
              {
                AssertThrow(false, ExcNotImplemented());
                break;
              }
            }
        }
    }



    template <int dim>
    void
    ReactiveFluidTransport<dim>::initialize()
    {
      base_model->initialize();
    }



    template <int dim>
    void
    ReactiveFluidTransport<dim>::update()
    {
      base_model->update();
    }



    template <int dim>
    void
    ReactiveFluidTransport<dim>::evaluate(const typename Interface<dim>::MaterialModelInputs &in,
                                          typename Interface<dim>::MaterialModelOutputs &out) const
    {
      base_model->evaluate(in,out);

      const unsigned int porosity_idx = this->introspection().compositional_index_for_name("porosity");

      // Modify the viscosity from the base model based on the presence of fluid.
      if (in.requests_property(MaterialProperties::viscosity))
        {
          // Scale the base model viscosity value based on the porosity.
          for (unsigned int q=0; q<out.n_evaluation_points(); ++q)
            {
              const double porosity = std::max(in.composition[q][porosity_idx],0.0);
              out.viscosities[q] *= (1.0 - porosity) * exp(- alpha_phi * porosity);
            }
        }

      // Fill the melt outputs if they exist. Note that the MeltOutputs class was originally
      // designed for two-phase flow material models in ASPECT that model the flow of melt,
      // but can be reused for a geofluid of arbitrary composition.
      MeltOutputs<dim> *fluid_out = out.template get_additional_output<MeltOutputs<dim>>();

      if (fluid_out != nullptr)
        {
          for (unsigned int q=0; q<out.n_evaluation_points(); ++q)
            {
              double porosity = std::max(in.composition[q][porosity_idx],0.0);

              fluid_out->fluid_viscosities[q] = eta_f;
              fluid_out->permeabilities[q] = reference_permeability * std::pow(porosity,3) * std::pow(1.0-porosity,2);

              double temperature_dependence; 
              if (fluid_solid_reaction_scheme == katz2003)
                {
                  // first, calculate temperature dependence of density
                  temperature_dependence = 1.0;
                  if (this->include_adiabatic_heating ())
                    {
                      // temperature dependence is 1 - alpha * (T - T(adiabatic))
                      temperature_dependence -= (in.temperature[q] - this->get_adiabatic_conditions().temperature(in.position[q]))
                                                * thermal_expansivity;
                    }
                  else
                    temperature_dependence -= (in.temperature[q] - reference_T) * thermal_expansivity;
                
                  // the fluid compressibility includes two parts, a constant compressibility, and a pressure-dependent one
                  // this is a simplified formulation, experimental data are often fit to the Birch-Murnaghan equation of state
                  const double fluid_compressibility_pressure = fluid_compressibility / (1.0 + in.pressure[q] * fluid_bulk_modulus_derivative * fluid_compressibility);
                  fluid_out->fluid_densities[q] = reference_rho_f * std::exp(fluid_compressibility_pressure * (in.pressure[q] - this->get_surface_pressure()))
                                             * temperature_dependence;

                  fluid_out->fluid_density_gradients[q] = fluid_out->fluid_densities[q] * fluid_out->fluid_densities[q]
                                                        * fluid_compressibility_pressure
                                                        * this->get_gravity_model().gravity_vector(in.position[q]);
                }else
                   fluid_out->fluid_densities[q] = reference_rho_f * std::exp(fluid_compressibility * (in.pressure[q] - this->get_surface_pressure()));

              if (in.requests_property(MaterialProperties::viscosity))
                {
                  const double phi_0 = 0.05;

                  // Limit the porosity to be no smaller than 1e-8 when
                  // calculating fluid effects on viscosities.
                  porosity = std::max(porosity,1e-8);
                  fluid_out->compaction_viscosities[q] = out.viscosities[q] * shear_to_bulk_viscosity_ratio * phi_0/porosity;
                  if (fluid_solid_reaction_scheme == katz2003)
                    {
                      double visc_temperature_dependence = 1.0;
                      if (this->include_adiabatic_heating ())
                        {
                          const double delta_temp = in.temperature[q]-this->get_adiabatic_conditions().temperature(in.position[q]);
                          visc_temperature_dependence = std::max(std::min(std::exp(-thermal_bulk_viscosity_exponent*delta_temp/this->get_adiabatic_conditions().temperature(in.position[q])),1e4),1e-4);
                        }
                      else
                        {
                          const double delta_temp = in.temperature[q]-reference_T;
                          const double T_dependence = (thermal_bulk_viscosity_exponent == 0.0
                                                      ?
                                                      0.0
                                                      :
                                                      thermal_bulk_viscosity_exponent*delta_temp/reference_T);
                          visc_temperature_dependence = std::max(std::min(std::exp(-T_dependence),1e4),1e-4);
                        }
                      fluid_out->compaction_viscosities[q] *= visc_temperature_dependence;
                    }
                }
            }
        }

      ReactionRateOutputs<dim> *reaction_rate_out = out.template get_additional_output<ReactionRateOutputs<dim>>();

      // Fill reaction rate outputs if the model uses operator splitting.
      // Specifically, change the porosity (representing the amount of free fluid)
      // based on the water solubility and the fluid content.
      if (this->get_parameters().use_operator_splitting && reaction_rate_out != nullptr)
        {
          std::vector<double> eq_free_fluid_fractions(out.n_evaluation_points());
          melt_fractions(in, eq_free_fluid_fractions);

          for (unsigned int q=0; q<out.n_evaluation_points(); ++q)
            for (unsigned int c=0; c<in.composition[q].size(); ++c)
              {

                // if (fluid_solid_reaction_scheme == katz2003)
                //   {
                //     // calculate density first, we need it for the reaction term
                //     // first, calculate temperature dependence of density
                //     double temperature_dependence = 1.0;
                //     if (this->include_adiabatic_heating ())
                //       {
                //         // temperature dependence is 1 - alpha * (T - T(adiabatic))
                //         temperature_dependence -= (in.temperature[q] - this->get_adiabatic_conditions().temperature(in.position[q]))
                //                                   * thermal_expansivity;
                //       }
                //     else
                //       temperature_dependence -= (in.temperature[q] - reference_T) * thermal_expansivity;

                //     // calculate composition dependence of density
                //     const double delta_rho = this->introspection().compositional_name_exists("peridotite")
                //                             ?
                //                             depletion_density_change * in.composition[q][this->introspection().compositional_index_for_name("peridotite")]
                //                             :
                //                             0.0;
                //     out.densities[q] = (reference_rho_s + delta_rho)
                //                       * temperature_dependence * std::exp(compressibility * (in.pressure[q] - this->get_surface_pressure()));
                //   }


                if (fluid_solid_reaction_scheme != katz2003)
                  {

                    double porosity_change = eq_free_fluid_fractions[q] - in.composition[q][porosity_idx];
                    // do not allow negative porosity
                    if (in.composition[q][porosity_idx] + porosity_change < 0)
                    porosity_change = -in.composition[q][porosity_idx];

                    const unsigned int bound_fluid_idx = this->introspection().compositional_index_for_name("bound_fluid");
                    if (c == bound_fluid_idx && this->get_timestep_number() > 0)
                      reaction_rate_out->reaction_rates[q][c] = - porosity_change / fluid_reaction_time_scale;
                    else if (c == porosity_idx && this->get_timestep_number() > 0)
                      reaction_rate_out->reaction_rates[q][c] = porosity_change / fluid_reaction_time_scale;
                    else
                      reaction_rate_out->reaction_rates[q][c] = 0.0;
                  }
                else
                  {
                     if (this->include_melt_transport())
                        {
                          const unsigned int peridotite_idx = this->introspection().compositional_index_for_name("peridotite");
                          const double old_porosity = in.composition[q][porosity_idx];
                          const double maximum_melt_fraction = in.composition[q][peridotite_idx];

                          // calculate the melting rate as difference between the equilibrium melt fraction
                          // and the solution of the previous time step
                          double porosity_change = 0.0;
                          if (fractional_melting)
                            {
                              // solidus is lowered by previous melting events (fractional melting)
                              const double solidus_change = (maximum_melt_fraction - old_porosity) * depletion_solidus_change;
                              const double eq_melt_fraction = katz2003_model.melt_fraction(in.temperature[q] - solidus_change, this->get_adiabatic_conditions().pressure(in.position[q]));
                              porosity_change = eq_melt_fraction - old_porosity;
                            }
                          else
                            {
                              // batch melting
                              porosity_change = katz2003_model.melt_fraction(in.temperature[q], this->get_adiabatic_conditions().pressure(in.position[q]))
                                                - std::max(maximum_melt_fraction, 0.0);
                              porosity_change = std::max(porosity_change, 0.0);

                              // freezing of melt below the solidus
                              
                                // If the porosity is larger than the equilibrium melt fraction, melt should freeze again.
                                // Because we do not track the melt composition, we have to use a workaround here for freezing of melt:
                                // We reduce the porosity until either it reaches the equilibrium melt fraction, or the depletion
                                // (peridotite field), which decreases as melt freezes, reaches the same value as the equilibrium
                                // melt fraction, whatever happens earlier. An exception is when the melt fraction is zero; in this case
                                // all melt should freeze.
                                const double eq_melt_fraction = katz2003_model.melt_fraction(in.temperature[q], this->get_adiabatic_conditions().pressure(in.position[q]));

                                // If the porosity change is not negative, there is no freezing, and the change in porosity
                                // is covered by the melting relation above.

                                // porosity reaches the equilibrium melt fraction:
                                const double porosity_change_wrt_melt_fraction = std::min(eq_melt_fraction - old_porosity - porosity_change,0.0);

                                // depletion reaches the equilibrium melt fraction:
                                const double porosity_change_wrt_depletion = std::min((eq_melt_fraction - std::max(maximum_melt_fraction, 0.0))
                                                                                      * (1.0 - old_porosity) / (1.0 - maximum_melt_fraction),0.0);
                                double freezing_amount = std::max(porosity_change_wrt_melt_fraction, porosity_change_wrt_depletion);

                                if (eq_melt_fraction == 0.0)
                                  freezing_amount = - old_porosity;

                                porosity_change += freezing_amount;

                                // Adapt time scale of freezing with respect to melting.
                                // We have to multiply with the melting time scale here to obtain the porosity change
                                // that happens in the time defined by the melting time scale (as opposed to a rate).
                                // This is important because we want to perform some checks on this quantity (for example,
                                // we want to make sure that this change does not lead to a negative porosity, see below).
                                // Later on, the overall porosity change is then divided again by the melting time scale
                                // to obtain the rate of melting or freezing, which is used in the operator splitting scheme.
                                if (porosity_change < 0 )
                                  porosity_change *= freezing_rate * fluid_reaction_time_scale;
                              
                            }

                          // remove melt that gets close to the surface
                          if (this->get_geometry_model().depth(in.position[q]) < extraction_depth)
                            porosity_change = -old_porosity * (in.position[q](1) - (this->get_geometry_model().maximal_depth() - extraction_depth))/extraction_depth;

                          // do not allow negative porosity
                          porosity_change = std::max(porosity_change, -old_porosity);

                          // because depletion is a volume-based, and not a mass-based property that is advected,
                          // additional scaling factors on the right hand side apply
                          // fill reaction rate outputs
                          if (reaction_rate_out != nullptr && in.requests_property(MaterialProperties::reaction_rates))
                            {
                              if (c == peridotite_idx && this->get_timestep_number() > 0)
                                reaction_rate_out->reaction_rates[q][c] = porosity_change / fluid_reaction_time_scale * (1 - maximum_melt_fraction) / (1 - old_porosity);
                              else if (c == porosity_idx && this->get_timestep_number() > 0)
                                reaction_rate_out->reaction_rates[q][c] = porosity_change / fluid_reaction_time_scale;
                              else
                                reaction_rate_out->reaction_rates[q][c] = 0.0;
                            }
                          out.reaction_terms[q][c] = 0.0;
                            

                          const double porosity = std::min(1.0, std::max(in.composition[q][porosity_idx],0.0));
                          out.viscosities[q] = eta_0 * exp(- alpha_phi * porosity);

                          out.entropy_derivative_pressure[q]    = katz2003_model.entropy_change (in.temperature[q], this->get_adiabatic_conditions().pressure(in.position[q]), maximum_melt_fraction, NonlinearDependence::pressure);
                          out.entropy_derivative_temperature[q] = katz2003_model.entropy_change (in.temperature[q], this->get_adiabatic_conditions().pressure(in.position[q]), maximum_melt_fraction, NonlinearDependence::temperature);
                        }
                      else
                        {
                          out.viscosities[q] = eta_0;

                          out.entropy_derivative_pressure[q]    = katz2003_model.entropy_change (in.temperature[q], this->get_adiabatic_conditions().pressure(in.position[q]), 0, NonlinearDependence::pressure);
                          out.entropy_derivative_temperature[q] = katz2003_model.entropy_change (in.temperature[q], this->get_adiabatic_conditions().pressure(in.position[q]), 0, NonlinearDependence::temperature);

                          // no melting/freezing is used in the model --> set all reactions to zero
                          out.reaction_terms[q][c] = 0.0;

                          if (reaction_rate_out != nullptr)
                              reaction_rate_out->reaction_rates[q][c] = 0.0;
                        }

                      out.thermal_expansion_coefficients[q] = thermal_expansivity;
                      out.specific_heat[q] = reference_specific_heat;
                      out.thermal_conductivities[q] = thermal_conductivity;
                      out.compressibilities[q] = compressibility;

                      double visc_temperature_dependence = 1.0;
                      if (this->include_adiabatic_heating ())
                        {
                          const double delta_temp = in.temperature[q]-this->get_adiabatic_conditions().temperature(in.position[q]);
                          visc_temperature_dependence = std::max(std::min(std::exp(-thermal_viscosity_exponent*delta_temp/this->get_adiabatic_conditions().temperature(in.position[q])),1e4),1e-4);
                        }
                      else
                        {
                          const double delta_temp = in.temperature[q]-reference_T;
                          const double T_dependence = (thermal_viscosity_exponent == 0.0
                                                      ?
                                                      0.0
                                                      :
                                                      thermal_viscosity_exponent*delta_temp/reference_T);
                          visc_temperature_dependence = std::max(std::min(std::exp(-T_dependence),1e4),1e-4);
                        }
                      out.viscosities[q] *= visc_temperature_dependence;
                    }
                  }
              }
        }
    }


    template <int dim>
    void
    ReactiveFluidTransport<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Reactive Fluid Transport Model");
        {


          prm.declare_entry("Base model","visco plastic",
                            Patterns::Selection(MaterialModel::get_valid_model_names_pattern<dim>()),
                            "The name of a material model incorporating the "
                            "addition of fluids. Valid values for this parameter "
                            "are the names of models that are also valid for the "
                            "``Material models/Model name'' parameter. See the documentation for "
                            "that for more information.");
          prm.declare_entry ("Reference fluid density", "2500",
                             Patterns::Double (0),
                             "Reference density of the melt/fluid$\\rho_{f,0}$. Units: \\si{\\kilogram\\per\\meter\\cubed}.");
          prm.declare_entry ("Shear to bulk viscosity ratio", "0.1",
                             Patterns::Double (0),
                             "Ratio between shear and bulk viscosity at the reference "
                             "permeability $\\phi_0=0.05$. The bulk viscosity additionally "
                             "scales with $\\phi_0/\\phi$. The shear viscosity is read in "
                             "from the base model. Units: dimensionless.");
          prm.declare_entry ("Reference fluid viscosity", "10",
                             Patterns::Double (0),
                             "The value of the constant melt/fluid viscosity $\\eta_f$. Units: \\si{\\pascal\\second}.");
          prm.declare_entry ("Exponential fluid weakening factor", "27",
                             Patterns::Double (0),
                             "The porosity dependence of the viscosity. Units: dimensionless.");
          prm.declare_entry ("Reference permeability", "1e-8",
                             Patterns::Double(),
                             "Reference permeability of the solid host rock."
                             "Units: \\si{\\meter\\squared}.");
          prm.declare_entry ("Fluid compressibility", "0.0",
                             Patterns::Double (0),
                             "The value of the compressibility of the fluid. "
                             "Units: \\si{\\per\\pascal}.");
          prm.declare_entry ("Fluid reaction time scale for operator splitting", "1e3",
                             Patterns::Double (0),
                             "In case the operator splitting scheme is used, the porosity field can not "
                             "be set to a new equilibrium fluid fraction instantly, but the model has to "
                             "provide a reaction time scale instead. This time scale defines how fast fluid "
                             "release and absorption happen, or more specifically, the parameter defines the "
                             "time after which the deviation of the porosity from the free fluid fraction "
                             "that would be in equilibrium with the solid will be reduced to a fraction of "
                             "$1/e$. So if the fluid reaction time scale is small compared "
                             "to the time step size, the reaction will be so fast that the porosity is very "
                             "close to this equilibrium value after reactions are computed. Conversely, "
                             "if the fluid reaction time scale is large compared to the time step size, almost no "
                             "fluid release and absorption will occur."
                             "\n\n"
                             "Also note that the fluid reaction time scale has to be larger than or equal to the reaction "
                             "time step used in the operator splitting scheme, otherwise reactions can not be "
                             "computed. If the model does not use operator splitting, this parameter is not used. "
                             "Units: yr or s, depending on the ``Use years "
                             "in output instead of seconds'' parameter.");
          prm.declare_entry ("Maximum weight percent water in sediment", "3",
                             Patterns::Double (0),
                             "The maximum allowed weight percent that the sediment composition can hold.");
          prm.declare_entry ("Maximum weight percent water in MORB", "2",
                             Patterns::Double (0),
                             "The maximum allowed weight percent that the sediment composition can hold.");
          prm.declare_entry ("Maximum weight percent water in gabbro", "1",
                             Patterns::Double (0),
                             "The maximum allowed weight percent that the sediment composition can hold.");
          prm.declare_entry ("Maximum weight percent water in peridotite", "8",
                             Patterns::Double (0),
                             "The maximum allowed weight percent that the sediment composition can hold.");
          prm.declare_entry ("Fluid-solid reaction scheme", "no reaction",
                             Patterns::Selection("no reaction|zero solubility|tian approximation|katz2003"),
                             "Select what type of scheme to use for reactions between fluid and solid phases. "
                             "The current available options are models where no reactions occur between "
                             "the two phases, or the solid phase is insoluble (zero solubility) and all "
                             "of the bound fluid is released into the fluid phase, tian approximation "
                             "use polynomials to describe hydration and dehydration reactions for four different "
                             "rock compositions as defined in Tian et al., 2019, or the Katz et. al. 2003 mantle "
                             "melting model.");
          prm.declare_entry ("Fluid bulk modulus derivative", "0.0",
                             Patterns::Double (0.),
                             "The value of the pressure derivative of the fluid bulk "
                             "modulus. "
                             "Units: None.");
          prm.declare_entry ("Thermal expansion coefficient", "2e-5",
                             Patterns::Double (0.),
                             "The value of the thermal expansion coefficient $\\beta$. "
                             "Units: \\si{\\per\\kelvin}.");
          prm.declare_entry ("Reference temperature", "293.",
                             Patterns::Double (0.),
                             "The reference temperature $T_0$. The reference temperature is used "
                             "in both the density and viscosity formulas. Units: \\si{\\kelvin}.");
          prm.declare_entry ("Thermal bulk viscosity exponent", "0.0",
                             Patterns::Double (0.),
                             "The temperature dependence of the bulk viscosity. Dimensionless exponent. "
                             "See the general documentation "
                             "of this model for a formula that states the dependence of the "
                             "viscosity on this factor, which is called $\\beta$ there.");
          prm.declare_entry ("Depletion density change", "0.0",
                             Patterns::Double (),
                             "The density contrast between material with a depletion of 1 and a "
                             "depletion of zero. Negative values indicate lower densities of "
                             "depleted material. Depletion is indicated by the compositional "
                             "field with the name peridotite. Not used if this field does not "
                             "exist in the model. "
                             "Units: \\si{\\kilogram\\per\\meter\\cubed}.");
          prm.declare_entry ("Depletion solidus change", "200.0",
                             Patterns::Double (0.),
                             "The solidus temperature change for a depletion of 100\\%. For positive "
                             "values, the solidus gets increased for a positive peridotite field "
                             "(depletion) and lowered for a negative peridotite field (enrichment). "
                             "Scaling with depletion is linear. Only active when fractional melting "
                             "is used. "
                             "Units: \\si{\\kelvin}.");
          prm.declare_entry ("Use fractional melting", "false",
                             Patterns::Bool (),
                             "If fractional melting should be used (if true), including a solidus "
                             "change based on depletion (in this case, the amount of melt that has "
                             "migrated away from its origin), and freezing of melt when it has moved "
                             "to a region with temperatures lower than the solidus; or if batch "
                             "melting should be used (if false), assuming that the melt fraction only "
                             "depends on temperature and pressure, and how much melt has already been "
                             "generated at a given point, but not considering movement of melt in "
                             "the melting parameterization."
                             "\n\n"
                             "Note that melt does not freeze unless the 'Freezing rate' parameter is set "
                             "to a value larger than 0.");
          prm.declare_entry ("Freezing rate", "0.0",
                             Patterns::Double (0.),
                             "Freezing rate of melt when in subsolidus regions. "
                             "If this parameter is set to a number larger than 0.0, it specifies the "
                             "fraction of melt that will freeze per year (or per second, depending on the "
                             "``Use years in output instead of seconds'' parameter), as soon as the porosity "
                             "exceeds the equilibrium melt fraction, and the equilibrium melt fraction "
                             "falls below the depletion. In this case, melt will freeze according to the "
                             "given rate until one of those conditions is not fulfilled anymore. The "
                             "reasoning behind this is that there should not be more melt present than "
                             "the equilibrium melt fraction, as melt production decreases with increasing "
                             "depletion, but the freezing process of melt also reduces the depletion by "
                             "the same amount, and as soon as the depletion falls below the equilibrium "
                             "melt fraction, we expect that material should melt again (no matter how "
                             "much melt is present). This is quite a simplification and not a realistic "
                             "freezing parameterization, but without tracking the melt composition, there "
                             "is no way to compute freezing rates accurately. "
                             "If this parameter is set to zero, no freezing will occur. "
                             "Note that freezing can never be faster than determined by the "
                             "``Melting time scale for operator splitting''. The product of the "
                             "``Freezing rate'' and the ``Melting time scale for operator splitting'' "
                             "defines how fast freezing occurs with respect to melting (if the "
                             "product is 0.5, melting will occur twice as fast as freezing). "
                             "Units: 1/yr or 1/s, depending on the ``Use years "
                             "in output instead of seconds'' parameter.");

          prm.declare_entry ("Melt extraction depth", "1000.0",
                             Patterns::Double (0.),
                             "Depth above that melt will be extracted from the model, "
                             "which is done by a negative reaction term proportional to the "
                             "porosity field. "
                             "Units: \\si{\\meter}.");

          prm.declare_entry ("Reference shear viscosity", "5e20",
                             Patterns::Double (0.),
                             "The value of the constant viscosity $\\eta_0$ of the solid matrix. "
                             "This viscosity may be modified by both temperature and porosity "
                             "dependencies. Units: \\si{\\pascal\\second}.");

          // read in melting model parameters
          ReactionModel::Katz2003MantleMelting<dim>::declare_parameters(prm);
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }

    template <int dim>
    void
    ReactiveFluidTransport<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Reactive Fluid Transport Model");
        {
          AssertThrow( prm.get("Base model") != "reactive fluid transport",
                       ExcMessage("You may not use ``reactive fluid transport'' "
                                  "as the base model for the reactive fluid transport "
                                  "model itself.") );

          reference_rho_f                   = prm.get_double ("Reference fluid density");
          shear_to_bulk_viscosity_ratio     = prm.get_double ("Shear to bulk viscosity ratio");
          eta_f                             = prm.get_double ("Reference fluid viscosity");
          reference_permeability            = prm.get_double ("Reference permeability");
          alpha_phi                         = prm.get_double ("Exponential fluid weakening factor");
          fluid_compressibility             = prm.get_double ("Fluid compressibility");
          fluid_reaction_time_scale         = prm.get_double ("Fluid reaction time scale for operator splitting");

          tian_max_peridotite_water         = prm.get_double ("Maximum weight percent water in peridotite");
          tian_max_gabbro_water             = prm.get_double ("Maximum weight percent water in gabbro");
          tian_max_MORB_water               = prm.get_double ("Maximum weight percent water in MORB");
          tian_max_sediment_water           = prm.get_double ("Maximum weight percent water in sediment");

          fluid_bulk_modulus_derivative     = prm.get_double ("Melt bulk modulus derivative");
          thermal_expansivity               = prm.get_double ("Thermal expansion coefficient");
          reference_T                       = prm.get_double ("Reference temperature");
          thermal_bulk_viscosity_exponent   = prm.get_double ("Thermal bulk viscosity exponent");
          depletion_density_change          = prm.get_double ("Depletion density change");
          depletion_solidus_change          = prm.get_double ("Depletion solidus change");
          fractional_melting                = prm.get_bool ("Use fractional melting");
          freezing_rate                     = prm.get_double ("Freezing rate");
          extraction_depth                  = prm.get_double ("Melt extraction depth");
          eta_0                             = prm.get_double ("Reference shear viscosity");

          // Create the base model and initialize its SimulatorAccess base
          // class; it will get a chance to read its parameters below after we
          // leave the current section.
          base_model = create_material_model<dim>(prm.get("Base model"));
          if (SimulatorAccess<dim> *sim = dynamic_cast<SimulatorAccess<dim>*>(base_model.get()))
            sim->initialize_simulator (this->get_simulator());

          if (this->convert_output_to_years() == true)
            fluid_reaction_time_scale *= year_in_seconds;

          // Reaction scheme parameter
          if (prm.get ("Fluid-solid reaction scheme") == "zero solubility")
            {
              fluid_solid_reaction_scheme = zero_solubility;
            }
          else if (prm.get ("Fluid-solid reaction scheme") == "no reaction")
            {
              fluid_solid_reaction_scheme = no_reaction;
            }
          else if (prm.get ("Fluid-solid reaction scheme") == "tian approximation")
            {
              AssertThrow(this->introspection().compositional_name_exists("sediment"),
                          ExcMessage("The Tian approximation only works "
                                     "if there is a compositional field called sediment."));
              AssertThrow(this->introspection().compositional_name_exists("MORB"),
                          ExcMessage("The Tian approximation only works "
                                     "if there is a compositional field called MORB."));
              AssertThrow(this->introspection().compositional_name_exists("gabbro"),
                          ExcMessage("The Tian approximation only works "
                                     "if there is a compositional field called gabbro."));
              AssertThrow(this->introspection().compositional_name_exists("peridotite"),
                          ExcMessage("The Tian approximation only works "
                                     "if there is a compositional field called peridotite."));
              fluid_solid_reaction_scheme = tian_approximation;
            }
          else if (prm.get ("Fluid-solid reaction scheme") == "katz2003")
            {
              fluid_solid_reaction_scheme = katz2003;
              katz2003_model.initialize_simulator (this->get_simulator());
              katz2003_model.parse_parameters(prm);
            }
          else
            AssertThrow(false, ExcMessage("Not a valid fluid-solid reaction scheme"));

          if (fluid_solid_reaction_scheme == no_reaction)
            {
              AssertThrow(this->get_parameters().use_operator_splitting == false,
                          ExcMessage("The Fluid-reaction scheme no reaction should not be used with operator splitting."));
            }

          if (fluid_solid_reaction_scheme == zero_solubility)
            {
              AssertThrow(this->get_parameters().use_operator_splitting,
                          ExcMessage("The Fluid-reaction scheme zero solubility must be used with operator splitting."));
            }

          if (fluid_solid_reaction_scheme == tian_approximation)
            {
              AssertThrow(this->get_parameters().use_operator_splitting,
                          ExcMessage("The Fluid-reaction scheme tian approximation must be used with operator splitting."));
            }

          if (this->get_parameters().use_operator_splitting)
            {
              AssertThrow(fluid_reaction_time_scale >= this->get_parameters().reaction_time_step,
                          ExcMessage("The reaction time step " + Utilities::to_string(this->get_parameters().reaction_time_step)
                                     + " in the operator splitting scheme is too large to compute fluid release rates! "
                                     "You have to choose it in such a way that it is smaller than the 'Fluid reaction time scale for "
                                     "operator splitting' chosen in the material model, which is currently "
                                     + Utilities::to_string(fluid_reaction_time_scale) + "."));
              AssertThrow(fluid_reaction_time_scale > 0,
                          ExcMessage("The Fluid reaction time scale for operator splitting must be larger than 0!"));
            }

          AssertThrow(this->introspection().compositional_name_exists("porosity"),
                      ExcMessage("Material model Reactive Fluid Transport only "
                                 "works if there is a compositional field called porosity."));

          if (fluid_solid_reaction_scheme != katz2003)
            {
              AssertThrow(this->introspection().compositional_name_exists("bound_fluid"),
                          ExcMessage("Material model Reactive Fluid Transport only "
                                     "works if there is a compositional field called bound_fluid."));
            }
          else
            {
              AssertThrow(this->introspection().compositional_name_exists("peridotite"),
                          ExcMessage("Material model Reactive Fluid Transport only "
                                     "works with the katz2003 model if there is a compositional "
                                     "field called peridotite."));
            }
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      // After parsing the parameters for this model, parse parameters related to the base model.
      base_model->parse_parameters(prm);
      this->model_dependence = base_model->get_model_dependence();
      if (fluid_solid_reaction_scheme == zero_solubility)
        {
          AssertThrow(this->get_material_model().is_compressible() == false,
                      ExcMessage("The Fluid-reaction scheme zero solubility must be used with an incompressible base model."));
        }
    }



    template <int dim>
    void
    ReactiveFluidTransport<dim>::create_additional_named_outputs (MaterialModel::MaterialModelOutputs<dim> &out) const
    {
      if (this->get_parameters().use_operator_splitting
          && out.template get_additional_output<ReactionRateOutputs<dim>>() == nullptr)
        {
          out.additional_outputs.push_back(
            std::make_unique<MaterialModel::ReactionRateOutputs<dim>> (out.n_evaluation_points(), this->n_compositional_fields()));
        }
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace MaterialModel
  {
    ASPECT_REGISTER_MATERIAL_MODEL(ReactiveFluidTransport,
                                   "reactive fluid transport",
                                   "Material model that is designed to advect fluids and compute "
                                   "fluid release and absorption based on different models for "
                                   "fluid-rock interaction. At present, models where no fluid-rock "
                                   "interactions occur or the solid has zero solubility are available. "
                                   "The properties of the solid can be taken from another material model "
                                   "that is used as a base model.")
  }
}
