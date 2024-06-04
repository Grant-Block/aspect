/*
  Copyright (C) 2015 - 2022 by the authors of the ASPECT code.

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


#include <aspect/material_model/melt_simple.h>
#include <aspect/material_model/reaction_model/katz2003_mantle_melting.h>
#include <aspect/utilities.h>
#include <aspect/adiabatic_conditions/interface.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/numerics/fe_field_function.h>


namespace aspect
{
  namespace MaterialModel
  {
    template <int dim>
    double
    MeltSimple<dim>::
    reference_darcy_coefficient () const
    {
      // 0.01 = 1% melt
      return katz2003_model.get_reference_permeability() * std::pow(0.01,3.0) / katz2003_model.get_eta_f();
    }

    template <int dim>
    bool
    MeltSimple<dim>::
    is_compressible () const
    {
      return model_is_compressible;
    }

    template <int dim>
    void
    MeltSimple<dim>::
    melt_fractions (const MaterialModel::MaterialModelInputs<dim> &in,
                    std::vector<double> &melt_fractions) const
    {
      for (unsigned int q=0; q<in.n_evaluation_points(); ++q)
        melt_fractions[q] = katz2003_model.melt_fraction(in.temperature[q],
                                                         this->get_adiabatic_conditions().pressure(in.position[q]));
    }


    template <int dim>
    void
    MeltSimple<dim>::initialize ()
    {
      if (this->include_melt_transport())
        {
          AssertThrow(this->get_parameters().use_operator_splitting,
                      ExcMessage("The material model ``Melt simple'' can only be used with operator splitting!"));
          AssertThrow(this->introspection().compositional_name_exists("peridotite"),
                      ExcMessage("Material model Melt simple only works if there is a "
                                 "compositional field called peridotite."));
          AssertThrow(this->introspection().compositional_name_exists("porosity"),
                      ExcMessage("Material model Melt simple with melt transport only "
                                 "works if there is a compositional field called porosity."));
        }
    }


    template <int dim>
    void
    MeltSimple<dim>::
    evaluate(const typename Interface<dim>::MaterialModelInputs &in, typename Interface<dim>::MaterialModelOutputs &out) const
    {
      katz2003_model.calculate_reaction_rate_outputs(in, out);
      katz2003_model.calculate_fluid_outputs(in, out);
    }


    template <int dim>
    void
    MeltSimple<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Melt simple");
        {
          // Melt Fraction Parameters
          ReactionModel::Katz2003MantleMelting<dim>::declare_parameters(prm);


          prm.declare_entry ("Use full compressibility", "false",
                             Patterns::Bool (),
                             "If the compressibility should be used everywhere in the code "
                             "(if true), changing the volume of material when the density changes, "
                             "or only in the momentum conservation and advection equations "
                             "(if false).");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    void
    MeltSimple<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Melt simple");
        {
          model_is_compressible      = prm.get_bool ("Use full compressibility");

          // Melt Fraction
          katz2003_model.initialize_simulator (this->get_simulator());
          katz2003_model.parse_parameters(prm);

        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }


    template <int dim>
    void
    MeltSimple<dim>::create_additional_named_outputs (MaterialModel::MaterialModelOutputs<dim> &out) const
    {
      if (this->get_parameters().use_operator_splitting && out.template get_additional_output<ReactionRateOutputs<dim>>() == nullptr)
        {
          const unsigned int n_points = out.n_evaluation_points();
          out.additional_outputs.push_back(
            std::make_unique<MaterialModel::ReactionRateOutputs<dim>> (n_points, this->n_compositional_fields()));
        }
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace MaterialModel
  {
    ASPECT_REGISTER_MATERIAL_MODEL(MeltSimple,
                                   "melt simple",
                                   "A material model that implements a simple formulation of the "
                                   "material parameters required for the modeling of melt transport, "
                                   "including a source term for the porosity according to the melting "
                                   "model for dry peridotite of \\cite{katz:etal:2003}. This also includes a "
                                   "computation of the latent heat of melting (if the `latent heat' "
                                   "heating model is active)."
                                   "\n\n"
                                   "Most of the material properties are constant, except for the shear, "
                                   "viscosity $\\eta$, the compaction viscosity $\\xi$, and the "
                                   "permeability $k$, which depend on the porosity; and the solid and melt "
                                   "densities, which depend on temperature and pressure:\n "
                                   "$\\eta(\\phi,T) = \\eta_0 e^{\\alpha(\\phi-\\phi_0)} e^{-\\beta(T-T_0)/T_0}$, "
                                   "$\\xi(\\phi,T) = \\xi_0 \\frac{\\phi_0}{\\phi} e^{-\\beta(T-T_0)/T_0}$, "
                                   "$k=k_0 \\phi^n (1-\\phi)^m$, "
                                   "$\\rho=\\rho_0 (1 - \\alpha (T - T_{\\text{adi}})) e^{\\kappa p}$."
                                   "\n\n"
                                   "The model is compressible only if this is specified in the input file, "
                                   "and contains compressibility for both solid and melt.")
  }
}
