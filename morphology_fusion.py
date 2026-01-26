"""
MORPHOLOGY FUSION ENGINE
========================================================================
Scientific integration of BET and XRD data for comprehensive
material characterization and journal recommendations.

References:
1. Sing et al., Pure Appl. Chem., 1985, 57, 603-619
2. Rouquerol et al., Adsorption by Powders and Porous Solids, 2014
3. Cullity & Stock, Elements of X-Ray Diffraction, 2001
========================================================================
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import warnings

warnings.filterwarnings('ignore')
# Add this import at the beginning of morphology_fusion.py
from scientific_integration import ScientificIntegrator
# ============================================================================
# MATERIAL CLASSIFICATION DATABASE
# ============================================================================
MATERIAL_DATABASE = {
    # Activated carbons
    'Activated Carbon': {
        'bet_range': (500, 3000),  # m²/g
        'pore_volume_range': (0.5, 2.0),  # cm³/g
        'micropore_fraction': (0.6, 0.9),
        'xrd_crystallinity': (0.0, 0.2),
        'crystal_system': 'Amorphous',
        'typical_applications': ['Adsorption', 'Catalysis', 'Energy Storage'],
        'journals': ['Carbon', 'Journal of Colloid and Interface Science', 'Microporous and Mesoporous Materials']
    },
    
    # Zeolites
    'Zeolite': {
        'bet_range': (200, 800),
        'pore_volume_range': (0.1, 0.4),
        'micropore_fraction': (0.8, 1.0),
        'xrd_crystallinity': (0.8, 1.0),
        'crystal_system': ['Cubic', 'Tetragonal', 'Orthorhombic'],
        'typical_applications': ['Catalysis', 'Ion Exchange', 'Gas Separation'],
        'journals': ['Microporous and Mesoporous Materials', 'Journal of Catalysis', 'Chemistry of Materials']
    },
    
    # MOFs
    'MOF': {
        'bet_range': (1000, 7000),
        'pore_volume_range': (0.5, 4.0),
        'micropore_fraction': (0.7, 1.0),
        'xrd_crystallinity': (0.9, 1.0),
        'crystal_system': ['Cubic', 'Tetragonal', 'Hexagonal'],
        'typical_applications': ['Gas Storage', 'Drug Delivery', 'Catalysis'],
        'journals': ['Journal of the American Chemical Society', 'Angewandte Chemie', 'Chemistry of Materials']
    },
    
    # Mesoporous silica
    'Mesoporous Silica': {
        'bet_range': (200, 1500),
        'pore_volume_range': (0.5, 2.0),
        'micropore_fraction': (0.0, 0.3),
        'xrd_crystallinity': (0.0, 0.3),
        'ordered_mesopores': True,
        'typical_applications': ['Drug Delivery', 'Catalysis', 'Chromatography'],
        'journals': ['Chemistry of Materials', 'Journal of Materials Chemistry', 'Microporous and Mesoporous Materials']
    },
    
    # Metal oxides
    'Metal Oxide': {
        'bet_range': (10, 200),
        'pore_volume_range': (0.05, 0.5),
        'micropore_fraction': (0.0, 0.3),
        'xrd_crystallinity': (0.5, 1.0),
        'crystal_system': ['Cubic', 'Tetragonal', 'Hexagonal', 'Monoclinic'],
        'typical_applications': ['Catalysis', 'Sensors', 'Energy Conversion'],
        'journals': ['Journal of Physical Chemistry C', 'Applied Catalysis B', 'ACS Catalysis']
    },
    
    # Carbon nanotubes
    'Carbon Nanotube': {
        'bet_range': (100, 500),
        'pore_volume_range': (0.3, 2.0),
        'micropore_fraction': (0.1, 0.4),
        'xrd_crystallinity': (0.7, 0.9),
        'crystal_system': 'Hexagonal',
        'typical_applications': ['Electronics', 'Composites', 'Energy Storage'],
        'journals': ['Carbon', 'ACS Nano', 'Nano Letters']
    },
    
    # Graphene oxides
    'Graphene Oxide': {
        'bet_range': (50, 500),
        'pore_volume_range': (0.1, 1.0),
        'micropore_fraction': (0.0, 0.2),
        'xrd_crystallinity': (0.3, 0.7),
        'crystal_system': 'Hexagonal',
        'typical_applications': ['Membranes', 'Composites', 'Energy Storage'],
        'journals': ['Carbon', 'ACS Nano', 'Advanced Materials']
    }
}

# ============================================================================
# HYSTERESIS TYPE INTERPRETATION
# ============================================================================
HYSTERESIS_INTERPRETATION = {
    'H1': {
        'description': 'Uniform cylindrical pores with narrow size distribution',
        'pore_shape': 'Cylindrical',
        'typical_materials': ['MCM-41', 'SBA-15', 'Ordered mesoporous materials'],
        'scientific_implications': 'High degree of pore ordering, suitable for size-selective applications'
    },
    'H2': {
        'description': 'Ink-bottle pores or interconnected pore network',
        'pore_shape': 'Ink-bottle or interconnected',
        'typical_materials': ['Some activated carbons', 'Porous glasses'],
        'scientific_implications': 'Complex pore connectivity, may show percolation effects'
    },
    'H3': {
        'description': 'Slit-shaped pores from plate-like particles',
        'pore_shape': 'Slit-shaped',
        'typical_materials': ['Clays', 'Graphite', 'Layered materials'],
        'scientific_implications': 'Anisotropic pore structure, often shows swelling behavior'
    },
    'H4': {
        'description': 'Combined micro-mesoporosity',
        'pore_shape': 'Mixed micro-mesoporous',
        'typical_materials': ['Activated carbons', 'Zeolite-templated carbons'],
        'scientific_implications': 'Hierarchical porosity, beneficial for mass transport'
    }
}

# ============================================================================
# MORPHOLOGY FUSION ENGINE
# ============================================================================
class MorphologyFusionEngine:
    """
    Engine for fusing BET and XRD data into comprehensive morphology analysis
    """
    
    def __init__(self):
        self.material_classification = []
        self.confidence_scores = {}
            # Add this line to initialize ScientificIntegrator
        self.scientific_integrator = ScientificIntegrator()
    def _classify_based_on_bet(self, bet_results: Dict) -> List[Dict]:
        """
        Classify material based on BET parameters
        """
        classifications = []
        
        S_bet = bet_results.get('surface_area', 0)
        V_total = bet_results.get('total_pore_volume', 0)
        V_micro = bet_results.get('micropore_volume', 0)
        
        if V_total > 0:
            micro_fraction = V_micro / V_total
        else:
            micro_fraction = 0
        
        # Check against material database
        for material, specs in MATERIAL_DATABASE.items():
            score = 0
            matched_criteria = []
            
            # Check surface area
            if 'bet_range' in specs:
                sa_min, sa_max = specs['bet_range']
                if sa_min <= S_bet <= sa_max:
                    score += 0.3
                    matched_criteria.append(f"Surface area ({S_bet:.0f} m²/g)")
            
            # Check pore volume
            if 'pore_volume_range' in specs:
                pv_min, pv_max = specs['pore_volume_range']
                if pv_min <= V_total <= pv_max:
                    score += 0.3
                    matched_criteria.append(f"Pore volume ({V_total:.2f} cm³/g)")
            
            # Check micropore fraction
            if 'micropore_fraction' in specs:
                mf_min, mf_max = specs['micropore_fraction']
                if mf_min <= micro_fraction <= mf_max:
                    score += 0.2
                    matched_criteria.append(f"Micropore fraction ({micro_fraction:.2f})")
            
            # Check hysteresis type
            hyst_type = bet_results.get('hysteresis_analysis', {}).get('type', '')
            if hyst_type and hyst_type in ['H1', 'H2']:
                if material in ['Mesoporous Silica', 'MOF']:
                    score += 0.2
                    matched_criteria.append(f"Hysteresis type {hyst_type}")
            
            if score > 0.3:  # Minimum threshold
                classifications.append({
                    'material': material,
                    'score': score,
                    'matched_criteria': matched_criteria,
                    'confidence': min(score, 1.0)
                })
        
        # Sort by score
        classifications.sort(key=lambda x: x['score'], reverse=True)
        
        return classifications[:3]  # Return top 3 matches
    
    def _classify_based_on_xrd(self, xrd_results: Dict) -> List[Dict]:
        """
        Classify material based on XRD parameters
        """
        classifications = []
        
        crystallinity = xrd_results.get('crystallinity_index', 0)
        crystal_system = xrd_results.get('crystal_system', 'Unknown')
        ordered_mesopores = xrd_results.get('ordered_mesopores', False)
        crystallite_size = xrd_results.get('crystallite_size', {}).get('scherrer', 0)
        
        # Check against material database
        for material, specs in MATERIAL_DATABASE.items():
            score = 0
            matched_criteria = []
            
            # Check crystallinity
            if 'xrd_crystallinity' in specs:
                ci_min, ci_max = specs['xrd_crystallinity']
                if ci_min <= crystallinity <= ci_max:
                    score += 0.4
                    matched_criteria.append(f"Crystallinity ({crystallinity:.2f})")
            
            # Check crystal system
            if 'crystal_system' in specs:
                crystal_spec = specs['crystal_system']
                if isinstance(crystal_spec, list):
                    if crystal_system in crystal_spec:
                        score += 0.3
                        matched_criteria.append(f"Crystal system ({crystal_system})")
                elif crystal_system == crystal_spec:
                    score += 0.3
                    matched_criteria.append(f"Crystal system ({crystal_system})")
            
            # Check ordered mesopores
            if ordered_mesopores and specs.get('ordered_mesopores', False):
                score += 0.3
                matched_criteria.append("Ordered mesopores")
            
            # Check crystallite size
            if crystallite_size > 0:
                if material == 'Metal Oxide' and crystallite_size < 50:
                    score += 0.1
                    matched_criteria.append(f"Crystallite size ({crystallite_size:.1f} nm)")
            
            if score > 0.3:  # Minimum threshold
                classifications.append({
                    'material': material,
                    'score': score,
                    'matched_criteria': matched_criteria,
                    'confidence': min(score, 1.0)
                })
        
        # Sort by score
        classifications.sort(key=lambda x: x['score'], reverse=True)
        
        return classifications[:3]  # Return top 3 matches
    
    def _fuse_classifications(self, bet_classes: List[Dict], 
                             xrd_classes: List[Dict]) -> Dict:
        """
        Fuse BET and XRD classifications
        """
        fused_results = {}
        
        # Combine classifications with weights
        all_materials = {}
        
        # Add BET classifications
        for cls in bet_classes:
            material = cls['material']
            all_materials[material] = {
                'bet_score': cls['score'],
                'xrd_score': 0,
                'bet_criteria': cls['matched_criteria'],
                'xrd_criteria': []
            }
        
        # Add XRD classifications
        for cls in xrd_classes:
            material = cls['material']
            if material in all_materials:
                all_materials[material]['xrd_score'] = cls['score']
                all_materials[material]['xrd_criteria'] = cls['matched_criteria']
            else:
                all_materials[material] = {
                    'bet_score': 0,
                    'xrd_score': cls['score'],
                    'bet_criteria': [],
                    'xrd_criteria': cls['matched_criteria']
                }
        
        # Calculate fused scores (weighted average)
        for material, scores in all_materials.items():
            # Weight: BET 60%, XRD 40% (can be adjusted)
            fused_score = (scores['bet_score'] * 0.6 + 
                         scores['xrd_score'] * 0.4)
            
            fused_results[material] = {
                'fused_score': fused_score,
                'bet_score': scores['bet_score'],
                'xrd_score': scores['xrd_score'],
                'bet_criteria': scores['bet_criteria'],
                'xrd_criteria': scores['xrd_criteria'],
                'confidence': fused_score
            }
        
        # Sort by fused score
        sorted_results = sorted(fused_results.items(), 
                              key=lambda x: x[1]['fused_score'], 
                              reverse=True)
        
        return dict(sorted_results[:5])  # Return top 5
    
    def _determine_dominant_feature(self, bet_results: Dict, 
                                   xrd_results: Dict) -> str:
        """
        Determine the dominant morphological feature
        """
        features = []
        
        # Surface area feature
        S_bet = bet_results.get('surface_area', 0)
        if S_bet > 1000:
            features.append(("Ultra-high surface area", 0.9))
        elif S_bet > 500:
            features.append(("High surface area", 0.7))
        elif S_bet > 100:
            features.append(("Moderate surface area", 0.5))
        else:
            features.append(("Low surface area", 0.3))
        
        # Porosity feature
        V_total = bet_results.get('total_pore_volume', 0)
        V_micro = bet_results.get('micropore_volume', 0)
        
        if V_total > 0:
            micro_fraction = V_micro / V_total
            
            if micro_fraction > 0.7:
                features.append(("Microporous dominant", 0.8))
            elif micro_fraction > 0.3:
                features.append(("Mixed micro-mesoporous", 0.7))
            else:
                features.append(("Mesoporous dominant", 0.6))
        
        # Crystallinity feature
        crystallinity = xrd_results.get('crystallinity_index', 0)
        if crystallinity > 0.8:
            features.append(("Highly crystalline", 0.9))
        elif crystallinity > 0.5:
            features.append(("Crystalline", 0.7))
        elif crystallinity > 0.2:
            features.append(("Semi-crystalline", 0.5))
        else:
            features.append(("Amorphous", 0.3))
        
        # Ordered structure feature
        if xrd_results.get('ordered_mesopores', False):
            features.append(("Ordered mesoporous structure", 0.8))
        
        # Hysteresis feature
        hyst_type = bet_results.get('hysteresis_analysis', {}).get('type', '')
        if hyst_type in HYSTERESIS_INTERPRETATION:
            hyst_desc = HYSTERESIS_INTERPRETATION[hyst_type]['description']
            features.append((f"{hyst_desc}", 0.7))
        
        # Find feature with highest weight
        if features:
            features.sort(key=lambda x: x[1], reverse=True)
            return features[0][0]
        
        return "Complex porous material"
    
    def _calculate_structure_property_relationships(self, 
                                                  bet_results: Dict,
                                                  xrd_results: Dict) -> List[str]:
        """
        Derive structure-property relationships
        """
        relationships = []
        
        # Surface area - activity relationship
        S_bet = bet_results.get('surface_area', 0)
        if S_bet > 1000:
            relationships.append("Ultra-high surface area suggests excellent potential for adsorption applications")
        elif S_bet > 500:
            relationships.append("High surface area indicates good catalytic potential")
        
        # Porosity - transport relationship
        V_micro = bet_results.get('micropore_volume', 0)
        V_total = bet_results.get('total_pore_volume', 0)
        
        if V_total > 0:
            micro_fraction = V_micro / V_total
            
            if micro_fraction > 0.7:
                relationships.append("Dominant microporosity suggests molecular sieving capability")
            elif micro_fraction > 0.3:
                relationships.append("Mixed porosity facilitates both adsorption and mass transport")
            else:
                relationships.append("Mesoporous dominant structure favors diffusion-limited processes")
        
        # Crystallinity - stability relationship
        crystallinity = xrd_results.get('crystallinity_index', 0)
        if crystallinity > 0.8:
            relationships.append("High crystallinity suggests good thermal and chemical stability")
        elif crystallinity < 0.3:
            relationships.append("Amorphous nature may provide flexibility and defect sites")
        
        # Ordered structure - reproducibility relationship
        if xrd_results.get('ordered_mesopores', False):
            relationships.append("Ordered mesoporous structure indicates precise synthetic control")
        
        return relationships
    
    def _suggest_applications(self, classifications: Dict,
                             bet_results: Dict) -> List[str]:
        """
        Suggest potential applications based on material properties
        """
        applications = []
        
        # Get top classification
        if classifications:
            top_material = list(classifications.keys())[0]
            material_specs = MATERIAL_DATABASE.get(top_material, {})
            
            # Add typical applications from database
            if 'typical_applications' in material_specs:
                applications.extend(material_specs['typical_applications'])
        
        # Add applications based on specific properties
        S_bet = bet_results.get('surface_area', 0)
        V_total = bet_results.get('total_pore_volume', 0)
        
        if S_bet > 800:
            if V_total > 1.0:
                applications.append("Gas storage (hydrogen, methane)")
                applications.append("Supercapacitor electrodes")
            else:
                applications.append("Catalyst support")
                applications.append("Molecular adsorption")
        
        if V_total > 0.8:
            applications.append("Drug delivery carrier")
            applications.append("Chromatographic separation")
        
        # Remove duplicates and limit
        unique_apps = list(dict.fromkeys(applications))
        return unique_apps[:6]  # Return top 6
    
    def _recommend_journals(self, classifications: Dict,
                           novelty_score: float) -> List[Dict]:
        """
        Recommend journals based on material type and novelty
        """
        recommendations = []
        
        # Get top classification
        if classifications:
            top_material = list(classifications.keys())[0]
            material_specs = MATERIAL_DATABASE.get(top_material, {})
            
            # Add journals from database
            if 'journals' in material_specs:
                for journal in material_specs['journals']:
                    recommendations.append({
                        'journal': journal,
                        'suitability': 'High',
                        'reason': f"Specializes in {top_material} materials"
                    })
        
        # Add general materials science journals
        general_journals = [
            ('Chemistry of Materials', 'High'),
            ('Journal of Materials Chemistry A', 'High'),
            ('ACS Applied Materials & Interfaces', 'Medium'),
            ('Materials Today', 'High'),
            ('Advanced Functional Materials', 'High')
        ]
        
        for journal, suitability in general_journals:
            recommendations.append({
                'journal': journal,
                'suitability': suitability,
                'reason': 'Broad materials science coverage'
            })
        
        # Sort by suitability
        recommendations.sort(key=lambda x: {'High': 0, 'Medium': 1, 'Low': 2}[x['suitability']])
        
        return recommendations[:8]  # Return top 8
    
    def _calculate_confidence_score(self, bet_results: Dict,
                                  xrd_results: Dict,
                                  classifications: Dict) -> float:
        """
        Calculate overall confidence score
        """
        confidence_factors = []
        
        # BET quality factor
        bet_r2 = bet_results.get('bet_regression', {}).get('r_squared', 0)
        confidence_factors.append(bet_r2 * 0.3)
        
        # XRD quality factor
        crystallinity = xrd_results.get('crystallinity_index', 0)
        n_peaks = len(xrd_results.get('peaks', []))
        xrd_quality = min(1.0, n_peaks / 20) * 0.5 + crystallinity * 0.5
        confidence_factors.append(xrd_quality * 0.3)
        
        # Classification consistency factor
        if classifications:
            top_score = list(classifications.values())[0]['fused_score']
            confidence_factors.append(top_score * 0.4)
        else:
            confidence_factors.append(0.2)
        
        return np.mean(confidence_factors)
    
    def _determine_novelty_factor(self, bet_results: Dict,
                                 xrd_results: Dict) -> float:
        """
        Determine novelty factor for the material
        """
        novelty_score = 0
        
        # Surface area novelty
        S_bet = bet_results.get('surface_area', 0)
        if S_bet > 2000:
            novelty_score += 0.4
        elif S_bet > 1000:
            novelty_score += 0.2
        
        # Porosity novelty
        V_total = bet_results.get('total_pore_volume', 0)
        if V_total > 2.0:
            novelty_score += 0.3
        
        # Structural novelty
        if xrd_results.get('ordered_mesopores', False):
            novelty_score += 0.2
        
        # Crystallinity combination
        crystallinity = xrd_results.get('crystallinity_index', 0)
        if crystallinity > 0.8 and S_bet > 1000:
            novelty_score += 0.1  # Rare combination
        
        return min(novelty_score, 1.0)
    
    def _recommend_techniques(self, bet_results: Dict,
                             xrd_results: Dict) -> List[str]:
        """
        Recommend further characterization techniques
        """
        techniques = []
        
        # Always recommend
        techniques.append("Thermogravimetric analysis (TGA) for thermal stability")
        techniques.append("Fourier-transform infrared spectroscopy (FTIR) for functional groups")
        
        # Based on properties
        S_bet = bet_results.get('surface_area', 0)
        if S_bet > 500:
            techniques.append("CO₂ adsorption for ultramicropore analysis")
        
        V_total = bet_results.get('total_pore_volume', 0)
        if V_total > 0.5:
            techniques.append("Mercury porosimetry for macropore analysis")
        
        if xrd_results.get('ordered_mesopores', False):
            techniques.append("Small-angle X-ray scattering (SAXS) for pore ordering")
            techniques.append("Transmission electron microscopy (TEM) for direct imaging")
        
        crystallinity = xrd_results.get('crystallinity_index', 0)
        if crystallinity > 0.7:
            techniques.append("Raman spectroscopy for defect analysis")
        
        return techniques[:6]
    
    def fuse(self, bet_results: Dict, xrd_results: Dict) -> Dict:
        """
        Main fusion method - combines BET and XRD data
        
        Parameters:
        -----------
        bet_results : BET analysis results
        xrd_results : XRD analysis results
        
        Returns:
        --------
        Fused morphology analysis
        """
        try:
            # Validate inputs
            if not bet_results.get('valid', False) or not xrd_results.get('valid', False):
                return {
                    'valid': False,
                    'error': 'Invalid input data',
                    'confidence_score': 0.0
                }
                    # Add scientific integration
            integration_results = self.scientific_integrator.integrate_results(bet_results, xrd_results)    
            
            # Step 1: Individual classifications
            bet_classifications = self._classify_based_on_bet(bet_results)
            xrd_classifications = self._classify_based_on_xrd(xrd_results)
            
            # Step 2: Fuse classifications
            fused_classifications = self._fuse_classifications(
                bet_classifications, xrd_classifications
            )
            
            # Step 3: Determine composite classification
            composite_classification = "Unknown"
            material_family = "Porous Material"
            
            if fused_classifications:
                top_material = list(fused_classifications.keys())[0]
                composite_classification = top_material
                
                # Group into families
                if any(m in top_material for m in ['Carbon', 'Graphene', 'Nanotube']):
                    material_family = "Carbon-based Material"
                elif 'Zeolite' in top_material or 'MOF' in top_material:
                    material_family = "Framework Material"
                elif 'Silica' in top_material:
                    material_family = "Silica-based Material"
                elif 'Oxide' in top_material:
                    material_family = "Metal Oxide"
            
            # Step 4: Calculate various metrics
            dominant_feature = self._determine_dominant_feature(bet_results, xrd_results)
            structure_relationships = self._calculate_structure_property_relationships(
                bet_results, xrd_results
            )
            suggested_applications = self._suggest_applications(
                fused_classifications, bet_results
            )
            journal_recommendations = self._recommend_journals(
                fused_classifications, 0.5
            )
            recommended_techniques = self._recommend_techniques(
                bet_results, xrd_results
            )
            confidence_score = self._calculate_confidence_score(
                bet_results, xrd_results, fused_classifications
            )
            novelty_factor = self._determine_novelty_factor(bet_results, xrd_results)
            
            # Calculate impact factor range based on novelty
            if novelty_factor > 0.7:
                impact_range = "8.0-15.0"
            elif novelty_factor > 0.4:
                impact_range = "5.0-8.0"
            else:
                impact_range = "3.0-5.0"
            
            # Calculate surface to volume ratio
            S_bet = bet_results.get('surface_area', 0)
            V_total = bet_results.get('total_pore_volume', 0)
            if V_total > 0:
                surface_to_volume = S_bet / (V_total * 1e6)  # Convert to m²/cm³
            else:
                surface_to_volume = 0
            
            # Calculate structural integrity index
            crystallinity = xrd_results.get('crystallinity_index', 0)
            n_peaks = len(xrd_results.get('peaks', []))
            structural_integrity = (crystallinity * 0.6 + 
                                  min(1.0, n_peaks/20) * 0.4)
            
            # Compile results
            results = {
                'valid': True,
                'composite_classification': composite_classification,
                'material_family': material_family,
                'dominant_feature': dominant_feature,
                'bet_classifications': bet_classifications,
                'xrd_classifications': xrd_classifications,
                'fused_classifications': fused_classifications,
                'structure_property_relationships': structure_relationships,
                'suggested_applications': suggested_applications,
                'journal_recommendations': journal_recommendations,
                'recommended_techniques': recommended_techniques,
                'confidence_score': confidence_score,
                'novelty_factor': novelty_factor,
                'impact_factor_range': impact_range,
                'surface_to_volume_ratio': surface_to_volume,
                'structural_integrity': structural_integrity,
                'hysteresis_interpretation': HYSTERESIS_INTERPRETATION.get(
                    bet_results.get('hysteresis_analysis', {}).get('type', ''),
                    {}
                ),
                'key_parameters': {
                    'surface_area_m2g': S_bet,
                    'pore_volume_cm3g': V_total,
                    'crystallinity_index': crystallinity,
                    'crystallite_size_nm': xrd_results.get('crystallite_size', {}).get('scherrer', 0)
                }
                'scientific_integration': integration_results
            }
        
            
            return results
            
        except Exception as e:
            return {
                'valid': False,
                'error': f"Fusion error: {str(e)}",
                'composite_classification': 'Unknown',
                'confidence_score': 0.0

            }
